import os
import json
import requests
from tqdm import tqdm
import yaml
import time
import shutil
import torch
import random
import datetime
import numpy as np
from torch.utils import data
from rootnav2.hourglass import hg
from rootnav2.loader import get_loader 
from rootnav2.utils import decode_segmap
from rootnav2.metrics import runningScore, averageMeter
from rootnav2.schedulers import get_scheduler
from rootnav2.optimizers import get_optimizer
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
import logging

# Loss functions
from monai.losses import DiceLoss
from rootnav2.loss.cldice_loss import HybridMultiClassCLDiceLoss
from rootnav2.loss import get_loss_function

# Metrics
from Metrics.gpu.haussdorff import HausdorffDistance
from Metrics.gpu.precision import Precision
from Metrics.gpu.recall import Recall
from Metrics.gpu.f1_score import F1Score
from Metrics.gpu.iou import MeanIoU
from Metrics.gpu.betti0_variation_index_gpu import Betti0VariationIndexGPU
from Metrics.gpu.betti1_variation_index_gpu import Betti1VariationIndexGPU
from Metrics.gpu.avg_centerline_distance import AverageCenterlineDistance
from Metrics.cpu.betti1_abs_error import Betti1AbsoluteError
from Metrics.cpu.betti0_abs_error import Betti0AbsoluteError
from Metrics.gpu.cldice import CLDice
from Metrics.gpu.dice import Dice
from Metrics.gpu.focal import FocalLoss
from Metrics.gpu.haussdorff_95 import HausdorffDistance95
from Metrics.gpu.mutual_information import NormalizedMutualInformation

weights = [0.0007,1.6246,0.7223,0.1789,1.748,12.9261] # [Background, Lateral Root, Primary Root, Seed, Primary Heatmap, Lateral Heatmap]

ALL_IDX = ["Background", "Lateral Root", "Primary Root", "Seed", "Primary Heatmap", "Lateral Heatmap"]
IDX_BACKGROUND = 0
IDX_LATERAL = 1
IDX_PRIMARY = 2

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
        "Hausdorff95": HausdorffDistance95(),
        "Precision": Precision(),
        "Recall": Recall(),
        "F1_Score": F1Score(),
        "Dice": Dice(),
        "CL_Dice": CLDice(),
        "Focal_loss": FocalLoss(),
        "IoU_Binary": MeanIoU(),
        "Betti0_abs": Betti0AbsoluteError(), # cpu
        "Betti1_abs": Betti1AbsoluteError(), # cpu
        "Normalized_Mutual_Info": NormalizedMutualInformation(),
        "CenterlineDist": AverageCenterlineDistance(threshold=0.5)
    }

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

    val_loss_seg_meter = averageMeter()
    val_loss_mse_meter = averageMeter()
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
            # loss_cldice = cldice_criterion(input=out_main, target=labels)
            
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
                writer.add_scalar('loss/train_loss_ce', loss1.item(), i+1)
                writer.add_scalar('loss/train_loss_mse', loss2.item(), i+1)
                time_meter.reset()

            if (i + 1) % cfg['training']['val_interval'] == 0 or (i + 1) == cfg['training']['train_iters']:
                model.eval()
                logger.info("Validation:")
                
                
                target_categories = {
                    'Lateral': IDX_LATERAL,       # 1
                    'Primary': IDX_PRIMARY        # 2
                }

                cat_meters = {
                    name: {k: averageMeter() for k in binary_metrics.keys()} 
                    for name in target_categories.keys()
                }
                cat_meters['Binary'] = {k: averageMeter() for k in binary_metrics.keys()}
                
                val_loss_seg_meter.reset()
                val_loss_mse_meter.reset()
                running_metrics_val.reset()
                
                with torch.no_grad():
                    val_loader_iter = tqdm(valloader, desc="Validation", unit="batch")
                    for images_val, labels_val, hm in val_loader_iter:
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        hm = hm.to(device)
                        
                        outputs = model(images_val)
                        outputs1 = outputs[-1]
                        
                        val_loss1 = ce_criterion(input=outputs1, target=labels_val)
                        val_loss_seg_meter.update(val_loss1.item())
                        val_loss2 = mse_criterion(input=torch.cat((outputs1[:,2:3,:,:], outputs1[:,4:5,:,:], outputs1[:,5:6,:,:]),1), target=hm)
                        val_loss_mse_meter.update(val_loss2.item())

                        val_loader_iter.set_postfix({'val_loss_seg': val_loss_seg_meter.avg, 'val_loss_mse': val_loss_mse_meter.avg})
                        pred_cls = outputs1.data.max(1)[1] # [Batch, H, W] (Indices 0-5)
                        pred_cls_np = pred_cls.cpu().numpy()
                        gt_np = labels_val.data.cpu().numpy()
                        running_metrics_val.update(gt_np, pred_cls_np)
                        
                        
                        batch_masks = {}
                       
                        # Background (0), Lateral (1), Primary (2)
                        for name, idx in target_categories.items():
                            pred_mask = (pred_cls == idx).float().unsqueeze(1)
                            gt_mask   = (labels_val == idx).float().unsqueeze(1)
                            batch_masks[name] = (pred_mask, gt_mask)

                        pred_comb = ((pred_cls == IDX_LATERAL) | (pred_cls == IDX_PRIMARY)).float().unsqueeze(1)
                        gt_comb   = ((labels_val == IDX_LATERAL) | (labels_val == IDX_PRIMARY)).float().unsqueeze(1)
                        batch_masks['Binary'] = (pred_comb, gt_comb)
                        
                        for cat_name, (p_mask, g_mask) in batch_masks.items():
                            for metric_name, metric_fn in binary_metrics.items():
                                try:
                                    val_loader_iter.set_postfix({f"{cat_name}_{metric_name}": f"computing"})
                                    val = metric_fn(p_mask, g_mask)
                                    if isinstance(val, torch.Tensor):
                                        val = val.item()
                                        
                                    if not np.isnan(val):
                                        cat_meters[cat_name][metric_name].update(val)
                                except Exception as e:
                                    pass

                val_loader_iter.close()
                writer.add_scalar('loss/val_loss_seg', val_loss_seg_meter.avg, i+1)
                writer.add_scalar('loss/val_loss_mse', val_loss_mse_meter.avg, i+1)
                score, class_iou = running_metrics_val.get_scores()
                
                cat_meters['Multi-Class'] = {}
                for key, value in score.items():
                    logger.info(f"{key}: {value:.4f}")
                    writer.add_scalar(f'val_multi_class/{key}', value, i+1)
                    cat_meters['Multi-Class'][key] = averageMeter()
                    cat_meters['Multi-Class'][key].update(value)
                    
                for key, value in class_iou.items():
                    logger.info(f"Class {key} IoU: {value:.4f}")
                    writer.add_scalar(f'val_class_iou/class_{key}_iou', value, i+1)
                    cat_meters['Multi-Class'][f'class_{key}_iou'] = averageMeter()
                    cat_meters['Multi-Class'][f'class_{key}_iou'].update(value)
                
                display_order = ['Primary', 'Lateral', 'Binary', "Multi-Class"]
                for cat in display_order:
                    logger.info(f"--- {cat} Segmentation Metrics ---")
                    for key, meter in cat_meters[cat].items():
                        logger.info(f"{key}: {meter.avg:.4f}")
                        writer.add_scalar(f'val_{cat}/{key}', meter.avg, i+1)

                logger.info("-----------------------------------")
                
                val_loss_seg_meter.reset()
                val_loss_mse_meter.reset()
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
                    pred1 = pred_cls[0].cpu().numpy()
                    channel_bindings = {'segmentation': {'Background': 0, 'Primary': 3, 'Lateral': 1}, 'heatmap': {'Seed': 5, 'Primary': 4, 'Lateral': 2}}
                    decoded = decode_segmap(np.array(pred1, dtype=np.uint8), channel_bindings)
                    decoded = Image.fromarray(decoded, 'RGBA')
                    example_path = os.path.join(logdir, 'validation_example.png')
                    decoded.save(example_path)
                    
                    from torchvision.utils import save_image
                    temp_dir = os.path.join(logdir, 'temp_viz')
                    os.makedirs(temp_dir, exist_ok=True)

                    img_tensor = images_val[0].cpu().float()
                    save_path_input = os.path.join(temp_dir, f'epoch_{i+1}_input.png')
                    save_image(img_tensor, save_path_input)
                    gt_numpy = labels_val[0].cpu().numpy().astype(np.uint8)
                    try:
                        decoded_gt = decode_segmap(gt_numpy, channel_bindings)
                        im_gt = Image.fromarray(decoded_gt, 'RGBA')
                        im_gt.save(os.path.join(temp_dir, f'epoch_{i+1}_ground_truth.png'))
                    except Exception as e:
                        logger.warning(f"Erreur lors du décodage du GT: {e}")

                    logger.info(f"Visualisation sauvegardée dans : {temp_dir}")

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break

    file_handler.close()