import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    """
    Hybrid loss with:
      - gradient-scale normalization (C)
      - exponential scheduling of stats contribution
      - efficient single-forward-pass warm-up
    L_total = alpha(n) * C * L_base + (1 - alpha(n)) * L_stats
    where alpha(n) = alpha_min + (1 - alpha_min) * 2^(-n / half_life_examples)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        stats_loss: nn.Module,
        alpha_min: float = 0.2,
        half_life_examples: int = 2000,
        max_examples_for_warmup: int = 500,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.stats_loss = stats_loss

        self.alpha_min = alpha_min
        self.half_life_examples = half_life_examples
        self.max_examples_for_warmup = max_examples_for_warmup
        self.epsilon = epsilon

        # Buffers (stateful, not trainable)
        self.register_buffer("C", torch.tensor(0.0))
        self.register_buffer("examples_seen", torch.tensor(0))

    def _compute_alpha(self) -> torch.Tensor:
        """
        Exponential schedule with half-life measured in examples.
        """
        n = self.examples_seen.float()
        hl = float(self.half_life_examples)
        return self.alpha_min + (1.0 - self.alpha_min) * torch.pow(
            torch.tensor(2.0, device=n.device), -n / hl
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = pred.shape[0]

        # -----------------------------
        # Compute base and stats losses
        # -----------------------------
        L_base = self.base_loss(pred, target)
        L_stats = self.stats_loss(pred, target)

        # ----------------------------------------
        # Warm-up phase: estimate C efficiently
        # ----------------------------------------
        if self.examples_seen < self.max_examples_for_warmup:
            # Compute gradients on a detached copy to avoid interfering with the main graph
            pred_for_grad = pred.detach().requires_grad_(True)
            L_base_for_grad = self.base_loss(pred_for_grad, target)
            L_stats_for_grad = self.stats_loss(pred_for_grad, target)

            grad_base = torch.autograd.grad(
                outputs=L_base_for_grad,
                inputs=pred_for_grad,
                create_graph=False
            )[0]
            grad_stats = torch.autograd.grad(
                outputs=L_stats_for_grad,
                inputs=pred_for_grad,
                create_graph=False
            )[0]

            C_current = grad_stats.norm() / (grad_base.norm() + self.epsilon)

            n_old = self.examples_seen
            n_new = n_old + batch_size
            self.C = (self.C * n_old + C_current * batch_size) / n_new

        # Always increment example counter
        self.examples_seen += batch_size

        # -----------------------------
        # Scheduled mixture
        # -----------------------------
        alpha = self._compute_alpha()
        total_loss = alpha * self.C * L_base + (1.0 - alpha) * L_stats

        return total_loss



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
    

class VarWeightedMSE(nn.Module):
    """
    Variance-weighted MSE for Poisson-distributed targets.
    Supports batched multi-channel inputs.
    
    Loss = mean( (pred - target)^2 / (k * target + epsilon) )
    """

    def __init__(self, k=1.0, epsilon=1e-8):
        """
        Parameters
        ----------
        k : float
            Scaling constant to convert activity (target) to counts
        epsilon : float
            Small constant to avoid divide-by-zero
        """
        super().__init__()
        self.k = k
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted activity, shape [B, C, H, W] or similar
        target : torch.Tensor
            Ground truth activity, same shape as pred
        """
        counts = self.k * target
        loss = ((pred - target) ** 2 / (counts + self.epsilon)).mean()
        return loss
