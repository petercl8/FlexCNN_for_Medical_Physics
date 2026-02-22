# Test comment: added by Copilot (2026-01-08)
import torch
import torch.nn as nn

class HybridLoss(nn.Module):
    """
    Hybrid loss combining a base loss and a statistics-based loss with dynamic weighting
    and gradient-scale normalization.
    
    This loss adaptively balances two loss components:
    - A base loss (typically pixel-wise, e.g., MSE or MAE)
    - A statistics loss (e.g., moment matching or perceptual loss)
    
    The combination uses two key mechanisms:
    1. **Gradient-scale normalization**: An EMA-tracked coefficient C that equalizes
       gradient magnitudes between the two losses, preventing one from dominating.
    2. **Exponential scheduling**: The mixing weight alpha(n) starts near 1.0 (favoring
       base loss early in training) and exponentially decays toward alpha_min, gradually
       incorporating the statistics loss.
    
    Formula
    -------
    L_total = alpha(n) * L_base + (1 - alpha(n)) * C * L_stats
    
    where:
        alpha(n) = alpha_min + (1 - alpha_min) * 2^(-n / half_life_examples)
        C = EMA of ||grad_L_base|| / ||grad_L_stats||
    
    Parameters
    ----------
    base_loss : nn.Module
        The primary loss function (e.g., nn.MSELoss(), nn.L1Loss()).
    stats_loss : nn.Module
        The statistics-based loss function (e.g., PatchwiseMomentLoss).
    alpha_min : float, default=0.2
        Minimum value for alpha scheduling. Training converges to this mixing ratio.
        Set to -1 to disable hybrid behavior and use only base_loss.
    half_life_examples : int, default=2000
        Number of examples for alpha to decay by half from initial value to alpha_min.
        Controls how quickly the stats loss is introduced.
    epsilon : float, default=1e-8
        Small constant to prevent division by zero in gradient normalization.
    show_components : bool, default=True
        If True, prints gradient norms and scaling factors during training for monitoring.
    C_momentum : float, default=0.05
        EMA momentum for updating the gradient scale coefficient C.
        Higher values adapt faster to changing gradient magnitudes.
    C_momentum_schedule : callable or None, default=None
        Optional function that takes examples_seen and returns a momentum value,
        allowing dynamic adjustment of EMA rate during training.
    
    Attributes
    ----------
    C : torch.Tensor
        EMA-tracked gradient scale normalization coefficient (buffer).
    examples_seen : torch.Tensor
        Counter for total examples processed, used for scheduling (buffer).
    
    Examples
    --------
    >>> base = nn.MSELoss()
    >>> stats = PatchwiseMomentLoss(patch_size=8, max_moment=2)
    >>> loss_fn = HybridLoss(base, stats, alpha_min=0.3, half_life_examples=1000)
    >>> pred = torch.randn(4, 1, 128, 128)
    >>> target = torch.randn(4, 1, 128, 128)
    >>> loss = loss_fn(pred, target)
    
    Notes
    -----
    - Setting alpha_min=-1 disables the hybrid mechanism and returns only base_loss.
    - The gradient normalization requires computing gradients internally, which adds
      computational overhead but ensures stable training.
    - Buffers (C and examples_seen) are automatically saved/loaded with model state.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        stats_loss: nn.Module,
        alpha_min: float = 0.2,
        half_life_examples: int = 2000,
        epsilon: float = 1e-8,
        show_components: bool = True,
        C_momentum: float = 0.05,
        C_momentum_schedule: callable = None,  # optional: function(examples_seen) -> momentum
    ):
        super().__init__()
        self.base_loss = base_loss
        self.stats_loss = stats_loss

        self.alpha_min = alpha_min
        self.half_life_examples = half_life_examples
        self.epsilon = epsilon
        self.show_components = show_components
        self.C_momentum = C_momentum
        self.C_momentum_schedule = C_momentum_schedule

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

        # Short-circuit to base loss only
        if self.alpha_min == -1:
            return L_base

        L_stats = self.stats_loss(pred, target)

        # -----------------------------
        # Estimate gradient norms for EMA
        # -----------------------------
        # Compute gradient norms with respect to predictions (detached)
        pred_for_grad = pred.detach().requires_grad_(True)
        L_base_for_grad = self.base_loss(pred_for_grad, target)
        L_stats_for_grad = self.stats_loss(pred_for_grad, target)

        grad_base = torch.autograd.grad(
            outputs=L_base_for_grad, inputs=pred_for_grad, create_graph=False
        )[0]
        grad_stats = torch.autograd.grad(
            outputs=L_stats_for_grad, inputs=pred_for_grad, create_graph=False
        )[0]

        g_base_norm = grad_base.norm()
        g_stats_norm = grad_stats.norm()
        C_current = g_base_norm / (g_stats_norm + self.epsilon)

        # Optional momentum schedule
        momentum = self.C_momentum
        if self.C_momentum_schedule is not None:
            momentum = self.C_momentum_schedule(self.examples_seen.item())

        # EMA update
        if self.examples_seen == 0:
            self.C = C_current
        else:
            self.C = (1.0 - momentum) * self.C + momentum * C_current

        if self.show_components:
            try:
                print(
                    f"[HybridLoss] ||grad_base||={g_base_norm.item():.6f}, "
                    f"||grad_stats||={g_stats_norm.item():.6f}, "
                    f"||grad_stats||*C={(g_stats_norm * self.C).item():.6f}"
                )
            except Exception:
                pass

        # -----------------------------
        # Scheduled mixture
        # -----------------------------
        alpha = self._compute_alpha()
        if self.show_components:
            try:
                print(f"[HybridLoss] alpha={alpha.item():.6f}, C={self.C.item():.6f}")
            except Exception:
                pass

        total_loss = alpha * L_base + (1.0 - alpha) * self.C * L_stats

        # Increment example counter
        self.examples_seen += batch_size

        return total_loss




class PatchwiseMomentLoss(nn.Module):
    """
    Patchwise statistical moment matching loss for texture and local statistics preservation.
    
    This loss extracts overlapping patches from predicted and target images, then computes
    and compares statistical moments (mean, variance, skewness, kurtosis, etc.) for each
    patch. It's particularly useful for preserving local texture characteristics beyond
    simple pixel-wise metrics.
    
    The loss computes central moments normalized by either mean or standard deviation to
    make them scale-invariant, then takes the absolute difference between predicted and
    target moments across all patches.
    
    Parameters
    ----------
    patch_size : int, default=8
        Side length of square patches to extract (in pixels).
    stride : int, default=4
        Stride for patch extraction. Smaller values create more overlapping patches.
    max_moment : int, default=3
        Maximum moment order to compute. Computes moments 1 through max_moment:
        - 1st moment: mean
        - 2nd moment: variance
        - 3rd moment: skewness
        - 4th moment: kurtosis, etc.
    scale : {'mean', 'std'}, default='mean'
        Normalization method for moments order k > 1:
        - 'mean': normalize by mean^k
        - 'std': normalize by std^k
    eps : float, default=1e-6
        Small constant to prevent division by zero in normalization.
    weights : list of float or None, default=None
        Weights for each moment order. If None, all moments weighted equally (1.0).
        Should have length equal to max_moment.
    
    Returns
    -------
    torch.Tensor
        Scalar loss value averaged over all patches, channels, and moment orders.
    
    Examples
    --------
    >>> # Match mean and variance in 8x8 patches
    >>> loss_fn = PatchwiseMomentLoss(patch_size=8, stride=4, max_moment=2)
    >>> pred = torch.randn(4, 1, 128, 128)
    >>> target = torch.randn(4, 1, 128, 128)
    >>> loss = loss_fn(pred, target)
    
    >>> # Emphasize higher-order moments with custom weights
    >>> loss_fn = PatchwiseMomentLoss(max_moment=4, weights=[0.5, 1.0, 1.5, 2.0])
    
    Notes
    -----
    - Larger patch_size captures more global statistics, smaller captures fine texture.
    - Smaller stride increases computational cost but provides more patch samples.
    - Higher-order moments (3rd, 4th) capture skewness and tail behavior.
    - This loss is complementary to pixel-wise losses and perceptual losses.
    """
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
    Variance-weighted mean squared error for Poisson-distributed data.
    
    This loss function accounts for the signal-dependent (heteroscedastic) noise inherent
    in photon-counting processes such as PET, SPECT, CT, and other medical imaging modalities.
    For Poisson statistics, variance equals the mean, so high-activity regions have higher
    noise variance. This loss weights errors inversely by the expected variance, giving
    more importance to low-noise (low-activity) regions.
    
    Formula
    -------
    Loss = mean( (pred - target)^2 / (k * target + epsilon) )
    
    where k converts activity units to photon counts.
    
    Parameters
    ----------
    k : float, default=1.0
        Scaling constant to convert activity values (target) to photon counts.
        For properly scaled data where target represents counts directly, use k=1.0.
        For normalized activity, set k to approximate counts per activity unit.
    epsilon : float, default=1e-8
        Small constant added to denominator to prevent division by zero in
        zero-activity regions.
    
    Returns
    -------
    torch.Tensor
        Scalar loss value averaged over all elements.
    
    Examples
    --------
    >>> # Standard usage with activity values
    >>> loss_fn = VarWeightedMSE(k=1.0, epsilon=1e-8)
    >>> pred = torch.rand(4, 1, 128, 128) * 100  # activity values
    >>> target = torch.rand(4, 1, 128, 128) * 100
    >>> loss = loss_fn(pred, target)
    
    >>> # For normalized data with known count scaling
    >>> loss_fn = VarWeightedMSE(k=1e5, epsilon=1e-8)  # ~100k counts per unit
    
    Notes
    -----
    - This loss is theoretically optimal for Poisson-distributed data under the assumption
      of maximum likelihood estimation.
    - Low-activity regions receive higher relative weight, which can be desirable for
      detecting small lesions but may amplify background artifacts.
    - Supports arbitrary-shaped tensors but expects pred and target to have the same shape.
    - The weighting is applied element-wise, so spatial structure is preserved.
    
    References
    ----------
    Related to weighted least squares and maximum likelihood estimation for Poisson data.
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


