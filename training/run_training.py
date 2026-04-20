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
import wandb

from torch.utils.tensorboard import SummaryWriter
import logging

# Loss functions
from monai.losses import DiceLoss
from rootnav2.loss.cldice_loss import HybridMultiClassCLDiceLoss

# Metrics
from training.Metrics.haussdorff import HausdorffDistance
from training.Metrics.precision import Precision
from training.Metrics.recall import Recall
from training.Metrics.specificity import Specificity
from training.Metrics.f1_score import F1Score
from training.Metrics.f_beta_score import FBetaScore
from training.Metrics.iou import IoU
from training.Metrics.mean_iou import MeanIoU
from training.Metrics.avg_centerline_distance import AverageSymetricCenterlineDistance
from training.Metrics.betti1_abs_err import Betti1AbsErrGPU
from training.Metrics.betti0_abs_err import Betti0AbsErrGPU
from training.Metrics.betti0_variation_index_gpu import Betti0VariationIndexGPU
from training.Metrics.betti1_variation_index_gpu import Betti1VariationIndexGPU
from training.Metrics.cldice import CLDice
from training.Metrics.dice import Dice
from training.Metrics.haussdorff_95 import HausdorffDistance95

weights = [0.0043, 1.2419, 0.837, 0.5495, 4.2761, 5.1519] # [Background, Lateral, Lateral Tip, Primary, Primary Tip, Seed] class weights computed with median frequency balancing on the training set. See

IDX_BACKGROUND = 0
IDX_LATERAL = 1
IDX_LATERAL_TIP = 2
IDX_PRIMARY = 3
IDX_PRIMARY_TIP = 4
IDX_SEED = 5

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
    wandb.init(
        entity='lgand-universit-de-montpellier',
        project="rootNav_logs",
        name=f"run_{run_id}_cfg_{os.path.basename(args.config)[:-4]}",
        config=cfg,
        #sync_tensorboard=True
    )
    
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
    t_loader = data_loader(data_path, split='train', hflip=hflip) #, network_input_size=None, network_output_size=None)
    v_loader = data_loader(data_path, split='valid') #, network_input_size=None, network_output_size=None)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=cfg['training']['batch_size'], num_workers=cfg['training']['n_workers'], shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=cfg['training']['batch_size'], num_workers=cfg['training']['n_workers'])

    running_metrics_val = runningScore(n_classes)

    binary_metrics = {
        "Hausdorff": HausdorffDistance(),
        "Hausdorff95": HausdorffDistance95(),
        "Precision": Precision(),
        "Recall": Recall(),
        "Specificity": Specificity(),
        "Dice": Dice(),
        "CLDice": CLDice(),
        "IoU": IoU(),
        "Mean_IoU": MeanIoU(),
        "Betti0_abs": Betti0AbsErrGPU(),
        "Betti1_abs": Betti1AbsErrGPU(),
        "Betti0_var_index": Betti0VariationIndexGPU(),
        "Betti1_var_index": Betti1VariationIndexGPU(),
        "ASCD": AverageSymetricCenterlineDistance(),
        "F1_Score": F1Score(),
        "F2_Score": FBetaScore(beta=2.0),
        "F3_Score": FBetaScore(beta=3.0),
        "F4_Score": FBetaScore(beta=4.0),
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
        
        start_iter = checkpoint.get("iter", checkpoint.get("epoch", 0))

    val_loss_seg_meter_ce = averageMeter()
    val_loss_seg_meter_dice = averageMeter()
    val_loss_seg_meter_cldice = averageMeter()
    val_loss_mse_meter = averageMeter()
    time_meter = averageMeter()
    best_iou = -100.0
    i = start_iter
    flag = True
    
    ##### LOSS FUNCTIONS #####
    ce_criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    dice_criterion = DiceLoss(
        #weight=class_weights, bad idea
        to_onehot_y=True,
        softmax=True, # transform logits to probs
        reduction='mean'
    ).to(device)
    
    cldice_criterion = HybridMultiClassCLDiceLoss(
        iter_=3, 
        alpha=0.5, 
        smooth=1.,
        root_indices=[IDX_LATERAL, IDX_PRIMARY],
    ).to(device)
        
    mse_criterion = torch.nn.MSELoss(reduction='mean').to(device)
    
    loss_name = cfg['training']['loss']['name']
    if loss_name == 'cross_entropy':
        logger.info("Using Cross Entropy Loss")
        loss = ce_criterion
    elif loss_name == 'dice':
        logger.info("Using Dice Loss")
        loss = dice_criterion
    elif loss_name == 'cldice_dice':
        logger.info("Using CLDice + Dice Loss")
        loss = cldice_criterion

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
            
            if loss_name == 'cross_entropy':
                loss1 = loss(input=out_main, target=labels)

            elif loss_name == 'dice':
                loss1 = loss(input=out_main, target=labels.unsqueeze(1))
            else:
                probs = torch.softmax(out_main, dim=1)
                target_one_hot = torch.nn.functional.one_hot(labels, num_classes=6)
                target_one_hot = target_one_hot.permute(0, 3, 1, 2).float().to(device)
                
                loss1 = loss(y_true=target_one_hot, y_pred=probs)
            
            out5 = out_main[:,5:6,:,:] 
            out4 = out_main[:,4:5,:,:]
            out2 = out_main[:,2:3,:,:] 
            tips = torch.cat((out2, out4,  out5), 1)
            loss2 = mse_criterion(input=tips, target=hm)
            
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
            
            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg['training']['print_interval'] == 0:
                logger.info("Iter [{:d}/{:d}] Loss: {:.4f} Time/Image: {:.4f}".format(
                    i + 1, cfg['training']['train_iters'], loss1.item(), time_meter.avg / cfg['training']['batch_size']))
                writer.add_scalar(f'loss/train_loss_seg_{loss_name}', loss1.item(), i+1)
                writer.add_scalar('loss/train_loss_mse', loss2.item(), i+1)
                
                wandb.log({
                    f"Train/Loss_Seg_{loss_name}": loss1.item(),
                    "Train/Loss_MSE": loss2.item(),
                }, step=i+1)
                
                time_meter.reset()

            if (i + 1) % cfg['training']['val_interval'] == 0 or (i + 1) == cfg['training']['train_iters']:
                model.eval()
                logger.info("Validation:")
                
                
                cat_meters = {
                    'Binary': {k: averageMeter() for k in binary_metrics.keys()}
                }
                                
                val_loss_seg_meter_ce.reset()
                val_loss_seg_meter_dice.reset()
                val_loss_seg_meter_cldice.reset()
                val_loss_mse_meter.reset()
                running_metrics_val.reset()
                
                with torch.inference_mode():
                    val_loader_iter = tqdm(valloader, desc="Validation", unit="batch")
                    for images_val, labels_val, hm in val_loader_iter:
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        hm = hm.to(device)
                        
                        outputs = model(images_val)
                        outputs1 = outputs[-1]
                        
                        val_loss_ce = ce_criterion(input=outputs1, target=labels_val)
                        val_loss_seg_meter_ce.update(val_loss_ce.item())
                        val_loss_dice = dice_criterion(input=outputs1, target=labels_val.unsqueeze(1))
                        val_loss_seg_meter_dice.update(val_loss_dice.item())
                        
                        probs_val = torch.softmax(outputs1, dim=1)
                        target_one_hot_val = torch.nn.functional.one_hot(labels_val, num_classes=6)
                        target_one_hot_val = target_one_hot_val.permute(0, 3, 1, 2).float()
                        val_loss_cldice = cldice_criterion(y_true=target_one_hot_val, y_pred=probs_val)
                        val_loss_seg_meter_cldice.update(val_loss_cldice.item())
                        val_loss2 = mse_criterion(input=torch.cat((outputs1[:,2:3,:,:], outputs1[:,4:5,:,:], outputs1[:,5:6,:,:]),1), target=hm)
                        val_loss_mse_meter.update(val_loss2.item())

                        val_loader_iter.set_postfix({'val_loss_seg_ce': val_loss_seg_meter_ce.avg, 'val_loss_seg_dice': val_loss_seg_meter_dice.avg, 'val_loss_seg_cldice': val_loss_seg_meter_cldice.avg, 'val_loss_mse': val_loss_mse_meter.avg})
                        pred_cls = outputs1.data.max(1)[1] # compute the argmax to get predicted class indices for each pixel
                        pred_cls_np = pred_cls.cpu().numpy()
                        gt_np = labels_val.data.cpu().numpy()
                        running_metrics_val.update(gt_np, pred_cls_np)
                        
                        
                        batch_masks = {}

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
                                    print(f"CRASH Metric {metric_name} sur {cat_name}: {e}")
                                    pass

                val_loader_iter.close()
                
                wandb_logs = {}
                
                writer.add_scalar('loss/val_loss_seg_ce', val_loss_seg_meter_ce.avg, i+1)
                writer.add_scalar('loss/val_loss_seg_dice', val_loss_seg_meter_dice.avg, i+1)
                writer.add_scalar('loss/val_loss_seg_cldice', val_loss_seg_meter_cldice.avg, i+1)
                writer.add_scalar('loss/val_loss_mse', val_loss_mse_meter.avg, i+1)
                
                wandb_logs["Val_Loss/CE"] = val_loss_seg_meter_ce.avg
                wandb_logs["Val_Loss/Dice"] = val_loss_seg_meter_dice.avg
                wandb_logs["Val_Loss/CLDice"] = val_loss_seg_meter_cldice.avg
                wandb_logs["Val_Loss/MSE"] = val_loss_mse_meter.avg
                
                score, class_iou = running_metrics_val.get_scores()
                
                cat_meters['Multi-Class'] = {}
                for key, value in score.items():
                    logger.info(f"{key}: {value:.4f}")
                    writer.add_scalar(f'val_multi_class/{key}', value, i+1)
                    
                    wandb_logs[f"Val_MultiClass/{key}"] = value
                    
                    cat_meters['Multi-Class'][key] = averageMeter()
                    cat_meters['Multi-Class'][key].update(value)
                    
                for key, value in class_iou.items():
                    logger.info(f"Class {key} IoU: {value:.4f}")
                    writer.add_scalar(f'val_class_iou/class_{key}_iou', value, i+1)
                    
                    wandb_logs[f"Val_Class_IoU/Class_{key}_IoU"] = value
                    
                    cat_meters['Multi-Class'][f'class_{key}_iou'] = averageMeter()
                    cat_meters['Multi-Class'][f'class_{key}_iou'].update(value)
                
                display_order = ['Binary', "Multi-Class"] # 'Primary', 'Lateral'
                for cat in display_order:
                    logger.info(f"--- {cat} Segmentation Metrics ---")
                    for key, meter in cat_meters[cat].items():
                        logger.info(f"{key}: {meter.avg:.4f}")
                        writer.add_scalar(f'val_{cat}/{key}', meter.avg, i+1)
                        
                        wandb_logs[f"Val_{cat}/{key}"] = meter.avg

                logger.info("-----------------------------------")
                wandb.log(wandb_logs, step=i+1)
                
                val_loss_seg_meter_ce.reset()
                val_loss_seg_meter_dice.reset()
                val_loss_seg_meter_cldice.reset()
                val_loss_mse_meter.reset()
                running_metrics_val.reset()

                current_real_epoch = (i + 1) // len(trainloader)
                
                state = {
                    "iter": i,
                    "epoch": current_real_epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_iou": best_iou,
                }

                save_name = "{}_{}_iter_{}.pkl".format(cfg['model']['arch'], cfg['data']['dataset'], i+1)
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
                    
                    wandb.log({"Validation/Prediction_Example": wandb.Image(example_path)}, step=i+1)

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break

    file_handler.close()
    wandb.finish()