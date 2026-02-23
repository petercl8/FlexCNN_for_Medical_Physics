"""
Custom loss functions for medical image reconstruction.

This module contains all custom loss functions including hybrid loss with gradient
normalization, patchwise moment matching for statistical regularization, and 
variance-weighted MSE for Poisson-distributed data.
"""

import torch
import torch.nn as nn
from typing import Callable

from . import defaults
from .core import _compute_patchwise_moments


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
    alpha_min : float
        Minimum value for alpha scheduling. Training converges to this mixing ratio.
        Set to -1 to disable hybrid behavior and use only base_loss.
    half_life_examples : int
        Number of examples for alpha to decay by half from initial value to alpha_min.
        Controls how quickly the stats loss is introduced.
    epsilon : float, default=defaults.HYBRID_EPSILON
        Small constant to prevent division by zero in gradient normalization.
    show_components : bool, default=defaults.HYBRID_SHOW_COMPONENTS
        If True, prints gradient norms and scaling factors during training for monitoring.
    C_momentum : float, default=defaults.HYBRID_C_MOMENTUM
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
    >>> stats = PatchwiseMomentLoss()
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
        - alpha_min and half_life_examples are required inputs and are set in trainables.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        stats_loss: nn.Module,
        alpha_min: float,
        half_life_examples: int,
        epsilon: float = defaults.HYBRID_EPSILON,
        show_components: bool = defaults.HYBRID_SHOW_COMPONENTS,
        C_momentum: float = defaults.HYBRID_C_MOMENTUM,
        C_momentum_schedule: Callable[[int], float] | None = None,
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
    
    This loss extracts overlapping patches from predicted and target images, computes
    statistical moments for each patch, and compares them between prediction and target.
    Optimized for PET reconstruction with physics-informed Poisson normalization, but also
    supports generic moment matching for arbitrary moment orders.
    
    Two operational modes:
    1. **PET Mode** (use_poisson_normalization=True, default): Physics-informed normalization
       - Mean differences normalized by patch mean (fractional bias)
       - Std differences normalized by sqrt(patch mean) (Poisson noise scale)
       - Only supports moments [1, 2]
    
    2. **Generic Mode** (use_poisson_normalization=False): Arbitrary moment matching
       - Supports moments [1, 2, 3, 4, ...] for texture analysis
       - Normalization by mean^k or std^k based on 'scale' parameter
    
    Parameters
    ----------
    patch_size : int, default=defaults.PATCH_SIZE
        Side length of square patches to extract (in pixels).
    stride : int, default=defaults.STRIDE
        Stride for patch extraction. Smaller values create more overlapping patches.
    moments : list of int, default=defaults.MOMENTS
        Which moments to compute. 1=mean, 2=variance/std, 3=skewness, 4=kurtosis, etc.
    moment_weights : dict or None, default=defaults.MOMENT_WEIGHTS
        Relative importance of each moment. Keys must match values in moments list.
        Example: {1: 0.5, 2: 1.0} weights std twice as much as mean.
        If None, all moments weighted equally (1.0).
    eps : float, default=defaults.EPS
        Small constant for numerical stability.
    patch_weighting : str, default=defaults.PATCH_WEIGHTING
        Patch importance weighting scheme:
        - 'scaled': Linear between patch_weight_min/max based on activity
        - 'energy': Weight by patch L2 energy
        - 'mean': Weight by patch mean activity
        - 'none': Uniform weighting
    patch_weight_min : float, default=defaults.PATCH_WEIGHT_MIN
        Minimum weight for 'scaled' mode.
    patch_weight_max : float, default=defaults.PATCH_WEIGHT_MAX
        Maximum weight for 'scaled' mode.
    max_patch_masked : float, default=defaults.MAX_PATCH_MASKED
        Mask patches with mean ≤ this threshold. Default 0 masks zero-activity patches.
    use_poisson_normalization : bool, default=defaults.USE_POISSON_NORMALIZATION
        Enable PET-specific Poisson normalization (requires moments ≤ 2).
    scale : str, default=defaults.SCALE
        Normalization for generic mode (use_poisson_normalization=False):
        - 'mean': normalize moment k by mean^k
        - 'std': normalize moment k by std^k
    counts_per_bq : float, default=defaults.COUNTS_PER_BQ
        Counts-per-activity scale for Poisson normalization.
    pathological_penalty : float, default=defaults.PATHOLOGICAL_PENALTY
        Large penalty value returned when predictions are pathological (NaN,
        negative mean activity, or all patches masked). Signals poor trial
        performance to Ray Tune without crashing, allowing scheduler to
        naturally deprioritize failing trials.
    
    Examples
    --------
    >>> # PET reconstruction (default, optimized for Poisson noise)
    >>> loss_fn = PatchwiseMomentLoss()
    >>> pred = torch.randn(4, 1, 128, 128)
    >>> target = torch.randn(4, 1, 128, 128)
    >>> loss = loss_fn(pred, target)
    
    >>> # Custom PET config with higher emphasis on std matching
    >>> loss_fn = PatchwiseMomentLoss(
    ...     moments=[1, 2],
    ...     moment_weights={1: 0.3, 2: 0.7},
    ...     patch_weighting='energy'
    ... )
    
    >>> # Generic texture matching with higher-order moments
    >>> loss_fn = PatchwiseMomentLoss(
    ...     moments=[1, 2, 3, 4],
    ...     use_poisson_normalization=False,
    ...     scale='std',
    ...     patch_weighting='none'
    ... )
    
    Notes
    -----
    - Default parameters are optimized for PET reconstruction
    - Patch weighting helps prioritize high-activity (high-SNR) regions
    - Low-activity masking prevents fitting noise in near-zero regions
    - Physics-informed normalization accounts for signal-dependent Poisson noise
    - Can be used in HybridLoss for dynamic weighting with pixel-wise losses
    
    See Also
    --------
    HybridLoss : Dynamically weighted combination of base and stats losses
    VarWeightedMSE : Variance-weighted MSE for Poisson data
    """
    
    def __init__(
        self,
        patch_size: int = defaults.PATCH_SIZE,
        stride: int = defaults.STRIDE,
        moments: list = defaults.MOMENTS,
        moment_weights: dict | None = defaults.MOMENT_WEIGHTS,
        eps: float = defaults.EPS,
        patch_weighting: str = defaults.PATCH_WEIGHTING,
        patch_weight_min: float = defaults.PATCH_WEIGHT_MIN,
        patch_weight_max: float = defaults.PATCH_WEIGHT_MAX,
        max_patch_masked: float = defaults.MAX_PATCH_MASKED,
        use_poisson_normalization: bool = defaults.USE_POISSON_NORMALIZATION,
        scale: str = defaults.SCALE,
        counts_per_bq: float = defaults.COUNTS_PER_BQ,
        pathological_penalty: float = defaults.PATHOLOGICAL_PENALTY
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.moments = moments
        self.moment_weights = moment_weights
        self.eps = eps
        self.patch_weighting = patch_weighting
        self.patch_weight_min = patch_weight_min
        self.patch_weight_max = patch_weight_max
        self.max_patch_masked = max_patch_masked
        self.use_poisson_normalization = use_poisson_normalization
        self.scale = scale
        self.counts_per_bq = counts_per_bq
        self.pathological_penalty = pathological_penalty

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute patchwise moment loss.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted images, shape [B, C, H, W]
        target : torch.Tensor
            Target images, same shape as pred
        
        Returns
        -------
        torch.Tensor
            Scalar loss value (differentiable)
        """
        loss, _ = _compute_patchwise_moments(
            pred=pred,
            target=target,
            moments=self.moments,
            moment_weights=self.moment_weights,
            patch_size=self.patch_size,
            stride=self.stride,
            eps=self.eps,
            patch_weighting=self.patch_weighting,
            patch_weight_min=self.patch_weight_min,
            patch_weight_max=self.patch_weight_max,
            max_patch_masked=self.max_patch_masked,
            use_poisson_normalization=self.use_poisson_normalization,
            scale=self.scale,
            counts_per_bq=self.counts_per_bq,
            pathological_penalty=self.pathological_penalty,
            return_penalty_for_pathological=False  # Loss needs real gradients, not penalty
        )
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
    k : float, default=defaults.COUNTS_PER_BQ
        Scaling constant to convert activity values (target) to photon counts.
        For properly scaled data where target represents counts directly, use k=1.0.
        For normalized activity, set k to approximate counts per activity unit.
        Default is 60.0 for PET activity maps.
    epsilon : float, default=defaults.VAR_WEIGHTED_EPSILON
        Small constant added to denominator to prevent division by zero in
        zero-activity regions.
    
    Returns
    -------
    torch.Tensor
        Scalar loss value averaged over all elements.
    
    Examples
    --------
    >>> # Standard usage with default PET parameters
    >>> loss_fn = VarWeightedMSE()
    >>> pred = torch.rand(4, 1, 128, 128) * 100
    >>> target = torch.rand(4, 1, 128, 128) * 100
    >>> loss = loss_fn(pred, target)
    
    >>> # For annihilation maps (counts represent actual photons)
    >>> loss_fn = VarWeightedMSE(k=1.0)
    
    >>> # For normalized data with known count scaling
    >>> loss_fn = VarWeightedMSE(k=1e5)  # ~100k counts per unit
    
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

    def __init__(
        self,
        k: float = defaults.COUNTS_PER_BQ,
        epsilon: float = defaults.VAR_WEIGHTED_EPSILON
    ):
        super().__init__()
        self.k = k
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute variance-weighted MSE.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted activity, shape [B, C, H, W] or similar
        target : torch.Tensor
            Ground truth activity, same shape as pred
        
        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        counts = self.k * target
        loss = ((pred - target) ** 2 / (counts + self.epsilon)).mean()
        return loss
