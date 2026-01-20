import os
import json
import requests
import yaml
import time
import shutil
import torch
import random
import datetime
import numpy as np
from torch.utils import data
from rootnav2.hourglass import hg
from rootnav2.loss import get_loss_function
from rootnav2.loader import get_loader 
from rootnav2.utils import decode_segmap
from rootnav2.metrics import runningScore, averageMeter
from rootnav2.schedulers import get_scheduler
from rootnav2.optimizers import get_optimizer
from PIL import Image
from monai.losses import DiceLoss
from rootnav2.loss.cldice_loss import HybridMultiClassCLDiceLoss


from torch.utils.tensorboard import SummaryWriter
import logging

from Metrics.gpu.haussdorff import HausdorffDistance
from Metrics.gpu.precision import Precision
from Metrics.gpu.recall import Recall
from Metrics.gpu.f1_score import F1Score
from Metrics.gpu.iou import MeanIoU
from Metrics.gpu.betti0_variation_index_gpu import Betti0VariationIndexGPU
from Metrics.gpu.betti1_variation_index_gpu import Betti1VariationIndexGPU
from Metrics.gpu.avg_centerline_distance import AverageCenterlineDistance

weights = [0.0007,1.6246,0.7223,0.1789,1.748,12.9261] # [Background, Lateral Root, Primary Root, Seed, Primary Heatmap, Lateral Heatmap]

def extract_url_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    try:
        return data['configuration']['network']['url']
    except KeyError:
        return None

def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    filename = url.split('/')[-1].replace(" ", "_")
    file_path = os.path.join(dest_folder, filename)
    r = requests.get(url, stream=True)
    if r.ok:
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())

def train(args):
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)

    run_id = random.randint(1,100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(run_id))
    
    writer = SummaryWriter(log_dir=logdir)
    logger = logging.getLogger()

    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_").replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    file_handler = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler) 

    if (args.debug):
        logger.setLevel(logging.DEBUG)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    shutil.copy(args.config, logdir)
    logger.info('Starting training')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.FloatTensor(weights).to(device)

    torch.manual_seed(cfg.get('seed', 65537))
    torch.cuda.manual_seed(cfg.get('seed', 65537))
    np.random.seed(cfg.get('seed', 65537))
    random.seed(cfg.get('seed', 65537))

    augmentations = cfg['training'].get('augmentations', 0.0)
    hflip = augmentations.get('hflip', 0.0) if augmentations is not None else 0.0

    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']
    t_loader = data_loader(data_path, split='train', hflip=hflip)
    v_loader = data_loader(data_path, split='valid')

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=cfg['training']['batch_size'], num_workers=cfg['training']['n_workers'], shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=cfg['training']['batch_size'], num_workers=cfg['training']['n_workers'])

    running_metrics_val = runningScore(n_classes)

    binary_metrics = {
        "Hausdorff": HausdorffDistance(),
        "Precision": Precision(),
        "Recall": Recall(),
        "F1_Score": F1Score(),
        "IoU_Binary": MeanIoU(),
        "Betti0_Var": Betti0VariationIndexGPU(),
        "Betti1_Var": Betti1VariationIndexGPU(),
        "CenterlineDist": AverageCenterlineDistance(threshold=0.5) # Peut être lent
    }
    binary_meters = {k: averageMeter() for k in binary_metrics.keys()}

    model = hg()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.to(device)

    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() if k != 'name'}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])
    #loss_fn = get_loss_function(cfg)

    start_iter = 0
    if cfg['training']['resume'] is not None and os.path.isfile(cfg['training']['resume']):
         checkpoint = torch.load(cfg['training']['resume'])
         model.load_state_dict(checkpoint["model_state"])
         optimizer.load_state_dict(checkpoint["optimizer_state"])
         scheduler.load_state_dict(checkpoint["scheduler_state"])
         start_iter = checkpoint["epoch"]

    val_loss_meter = averageMeter()
    time_meter = averageMeter()
    best_iou = -100.0
    i = start_iter
    flag = True
    
    ##### LOSS FUNCTIONS #####
    ce_criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    #dice_criterion = DiceLoss(
    #    weight=class_weights,
    #    to_onehot_y=True,
    #    softmax=True,
    #    reduction='mean'
    #).to(device)
    
    #cldice_criterion = HybridMultiClassCLDiceLoss(iter_=10, alpha=0.5, smooth=1., weights=None, root_indices=None, reduction='mean').to(device)
    
    mse_criterion = torch.nn.MSELoss(reduction='mean').to(device)

    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels, hm) in trainloader:
            i += 1
            start_ts = time.time()
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            hm = hm.to(device)

            outputs = model(images)
            out_main = outputs[-1]
            
            optimizer.zero_grad()
            loss1 = ce_criterion(input=out_main, target=labels)
            # loss_dice = dice_criterion(input=out_main, target=labels)
            
            out5 = out_main[:,5:6,:,:] 
            out4 = out_main[:,4:5,:,:]
            out2 = out_main[:,2:3,:,:] 
            tips = torch.cat((out2, out4,  out5), 1)
            loss2 = mse_criterion(input=tips, target=hm)

            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
            scheduler.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg['training']['print_interval'] == 0:
                logger.info("Iter [{:d}/{:d}] Loss: {:.4f} Time/Image: {:.4f}".format(
                    i + 1, cfg['training']['train_iters'], loss1.item(), time_meter.avg / cfg['training']['batch_size']))
                writer.add_scalar('loss/train_loss', loss1.item(), i+1)
                time_meter.reset()

            if (i + 1) % cfg['training']['val_interval'] == 0 or (i + 1) == cfg['training']['train_iters']:
                model.eval()
                logger.info("Validation:")
                
                for meter in binary_meters.values():
                    meter.reset()

                with torch.no_grad():
                    for images_val, labels_val, hm in valloader:
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        hm = hm.to(device)
                        
                        outputs = model(images_val)
                        outputs1 = outputs[-1]
                        
                        val_loss1 = ce_criterion(input=outputs1, target=labels_val)
                        val_loss_meter.update(val_loss1.item())

                        pred_cls = outputs1.data.max(1)[1] # [Batch, H, W] (Indices 0-5)
                        pred_cls_np = pred_cls.cpu().numpy()
                        gt_np = labels_val.data.cpu().numpy()
                        running_metrics_val.update(gt_np, pred_cls_np)

                        bin_pred = (pred_cls > 0).float().unsqueeze(1) # [B, 1, H, W]
                        bin_mask = (labels_val > 0).float().unsqueeze(1) # [B, 1, H, W]

                        for key, metric_fn in binary_metrics.items():
                            try:
                                val = metric_fn(bin_pred, bin_mask)
                                if not np.isnan(val):
                                    binary_meters[key].update(val)
                            except Exception as e:
                                logger.warning(f"Error calculating {key}: {e}")

                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i+1)
                score, class_iou = running_metrics_val.get_scores()

                logger.info("--- Binary Segmentation Metrics ---")
                for key, meter in binary_meters.items():
                    writer.add_scalar(f'val_metrics_binary/{key}', meter.avg, i+1)
                    logger.info(f"{key}: {meter.avg:.4f}")
                logger.info("-----------------------------------")

                logger.info(f"Overall Acc: {score['oacc']:.6f} | Mean IoU: {score['miou']:.6f}")

                val_loss_meter.reset()
                running_metrics_val.reset()

                state = {
                    "epoch": i + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_iou": best_iou,
                }
                
                save_name = "{}_{}_epoch_{}.pkl".format(cfg['model']['arch'], cfg['data']['dataset'], i + 1)
                save_path = os.path.join(logdir, save_name)
                torch.save(state, save_path)
                logger.info(f"Checkpoint saved: {save_path}")

                if score['miou'] >= best_iou:
                    best_iou = score['miou']
                    state["best_iou"] = best_iou
                    save_path_best = os.path.join(logdir, "{}_{}_best_model.pkl".format(cfg['model']['arch'], cfg['data']['dataset']))
                    torch.save(state, save_path_best)
                    logger.info(f"New Best Model saved (IoU: {best_iou:.4f})")

                if (args.output_example):
                    pred1 = np.squeeze(pred_cls_np[0], axis=0) # [H, W]
                    channel_bindings = {'segmentation': {'Background': 0, 'Primary': 3, 'Lateral': 1}, 'heatmap': {'Seed': 5, 'Primary': 4, 'Lateral': 2}}
                    decoded = decode_segmap(np.array(pred1, dtype=np.uint8), channel_bindings)
                    decoded = Image.fromarray(decoded, 'RGBA')
                    example_path = os.path.join(logdir, 'validation_example.png')
                    decoded.save(example_path)

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break

    file_handler.close()