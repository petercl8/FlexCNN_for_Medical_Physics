import torch
import torch.nn as nn

class PatchwiseMomentLoss(nn.Module):
    '''
    Implement by adding the following to appropriate loss entires in the search dicts:
        PatchwiseMomentLoss(patch_size=patch_size, stride=stride, max_moment=max_moment, scale=scale)  
    '''
    def __init__(self, patch_size=8, stride=4, max_moment=3, 
                 scale='mean', eps=1e-6, weights=None):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.max_moment = max_moment  # compute moments 1..max_moment
        self.scale = scale  # 'mean' or 'std'
        self.eps = eps
        self.weights = weights if weights is not None else [1.0]*max_moment

    def forward(self, pred, target):
        # pred, target: [B, C, H, W]
        # Extract overlapping patches
        B,C,H,W = pred.shape
        p = self.patch_size
        s = self.stride
        pred_patches = pred.unfold(2,p,s).unfold(3,p,s).contiguous().view(B,C,-1,p*p)
        target_patches = target.unfold(2,p,s).unfold(3,p,s).contiguous().view(B,C,-1,p*p)

        loss = 0.0
        for k in range(1, self.max_moment+1):
            # compute k-th central moment
            target_mean = target_patches.mean(dim=-1, keepdim=True)
            pred_mean = pred_patches.mean(dim=-1, keepdim=True)
            if k == 1:
                # first moment: mean loss
                moment_loss = torch.mean(torch.abs(pred_mean - target_mean))
            else:
                pred_c = pred_patches - pred_mean
                target_c = target_patches - target_mean
                pred_m = (pred_c ** k).mean(dim=-1)
                target_m = (target_c ** k).mean(dim=-1)

                if self.scale == 'std':
                    sigma = torch.sqrt((target_c**2).mean(dim=-1) + self.eps)
                    pred_m = pred_m / (sigma**k + self.eps)
                    target_m = target_m / (sigma**k + self.eps)
                elif self.scale == 'mean':
                    mean_val = target_mean.squeeze(-1) + self.eps
                    pred_m = pred_m / (mean_val**k)
                    target_m = target_m / (mean_val**k)

                moment_loss = torch.mean(torch.abs(pred_m - target_m))

            loss += self.weights[k-1] * moment_loss

        return loss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.moment_loss = PatchwiseMomentLoss(...)
        self.pixel_loss = nn.L1Loss()
    def forward(self, pred, target):
        return self.alpha*self.pixel_loss(pred, target) + (1-self.alpha)*self.moment_loss(pred, target)
