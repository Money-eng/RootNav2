import torch
import torch.nn as nn
from .cldice import soft_dice, soft_dice_cldice

class ProjectedCLDiceLoss(nn.Module): # Very bad
    def __init__(self, iter_=3, alpha=0.5, smooth=1., exclude_background=False):
        super(ProjectedCLDiceLoss, self).__init__()
        self.iter = iter_
        self.alpha = alpha
        self.smooth = smooth
        
        self.groups = {
            'roots': [1, 2],
            'seed': [3],
            'tips': [4, 5] 
        }
        
        self.cldice_fn = soft_dice_cldice(iter_=iter_, alpha=alpha, smooth=smooth, exclude_background=exclude_background)

    def forward(self, y_true, y_pred):
        pred_channels = []
        true_channels = []

        for _, indices in self.groups.items():
            p_proj = y_pred[:, indices, :, :].sum(dim=1, keepdim=True) # debatable
            t_proj = y_true[:, indices, :, :].sum(dim=1, keepdim=True)
            
            pred_channels.append(p_proj)
            true_channels.append(t_proj)

        y_pred_projected = torch.cat(pred_channels, dim=1)
        y_true_projected = torch.cat(true_channels, dim=1)
        
        y_pred_projected = torch.clamp(y_pred_projected, 0, 1)
        y_true_projected = torch.clamp(y_true_projected, 0, 1)

        loss = self.cldice_fn(y_true_projected, y_pred_projected)
        return loss

class HybridMultiClassCLDiceLoss(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1., weights=None, root_indices=None, reduction='mean'):
        super(HybridMultiClassCLDiceLoss, self).__init__()
        self.iter = iter_
        self.alpha = alpha
        self.smooth = smooth
        self.weights = weights
        self.root_indices = root_indices if root_indices is not None else []
        self.reduction = reduction
        
        self.cldice_fn = soft_dice_cldice(iter_=iter_, alpha=alpha, smooth=smooth, exclude_background=False)

    def forward(self, y_true, y_pred):

        n_classes = y_pred.shape[1]
        
        if self.weights is None:
            weights = torch.ones(n_classes, device=y_pred.device)
        else:
            weights = torch.tensor(self.weights, device=y_pred.device)
            
        loss_channels = []

        for i in range(n_classes):
            y_true_c = y_true[:, i:i+1]
            y_pred_c = y_pred[:, i:i+1]
            
            if i in self.root_indices:
                # Roots
                loss_c = self.cldice_fn(y_true_c, y_pred_c)
            else:
                # Rest
                loss_c = soft_dice(y_true_c, y_pred_c)
                
            loss_channels.append(loss_c)

        f = torch.stack(loss_channels)

        if f.ndim > 1:
             weights = weights.view(-1, 1)
        
        f = f * weights

        if self.reduction == 'mean':
            return torch.mean(f)
        elif self.reduction == 'sum':
            return torch.sum(f)
        else:
            return f