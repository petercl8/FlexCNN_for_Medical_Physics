"""
Evaluation metrics for statistical matching.

This module contains evaluation metrics that wrap the shared core computation
for use during model evaluation and validation.
"""

from . import defaults
from .core import _compute_patchwise_moments


def patchwise_moment_metric(
    batch_pred,
    batch_target,
    moments=defaults.MOMENTS,
    moment_weights=defaults.MOMENT_WEIGHTS,
    patch_size=defaults.PATCH_SIZE,
    stride=defaults.STRIDE,
    eps=defaults.EPS,
    patch_weighting=defaults.PATCH_WEIGHTING,
    patch_weight_min=defaults.PATCH_WEIGHT_MIN,
    patch_weight_max=defaults.PATCH_WEIGHT_MAX,
    max_patch_masked=defaults.MAX_PATCH_MASKED,
    use_poisson_normalization=defaults.USE_POISSON_NORMALIZATION,
    scale=defaults.SCALE,
    return_per_moment=False
):
    """
    Patchwise moment metric for PET reconstructions with physics-informed normalization.

    This function wraps the shared core computation (_compute_patchwise_moments) 
    for evaluation purposes, converting tensor output to scalar and optionally 
    returning per-moment breakdown.

    -----------------------------
    Physics-informed normalization:
    - Mean differences normalized by patch mean (fractional bias)
    - Std differences normalized by sqrt(patch mean) (Poisson noise scale)
    -----------------------------
    
    Parameters
    -----------
    batch_pred : torch.Tensor
        Predicted batch, shape [B, C, H, W]
    batch_target : torch.Tensor
        Target batch, same shape as batch_pred
    moments : list, default=defaults.MOMENTS
        Which moments to compute: 1=mean, 2=std
    moment_weights : dict or None, default=defaults.MOMENT_WEIGHTS
        Relative importance of each moment in final metric.
        If None, all moments weighted equally.
    patch_size : int, default=defaults.PATCH_SIZE
        Size of square patch
    stride : int, default=defaults.STRIDE
        Stride between patches
    eps : float, default=defaults.EPS
        Small constant for numerical stability
    patch_weighting : str, default=defaults.PATCH_WEIGHTING
        How to weight patches: 'scaled', 'energy', 'mean', or 'none'
    patch_weight_min : float, default=defaults.PATCH_WEIGHT_MIN
        Minimum weight when using 'scaled' weighting
    patch_weight_max : float, default=defaults.PATCH_WEIGHT_MAX
        Maximum weight when using 'scaled' weighting
    max_patch_masked : float, default=defaults.MAX_PATCH_MASKED
        Ignore patches with mean ≤ this threshold (default 0 = only zero patches)
    use_poisson_normalization : bool, default=defaults.USE_POISSON_NORMALIZATION
        Enable PET-specific Poisson normalization (requires moments ≤ 2)
    scale : str, default=defaults.SCALE
        Normalization for generic mode (use_poisson_normalization=False):
        - 'mean': normalize moment k by mean^k
        - 'std': normalize moment k by std^k
    return_per_moment : bool, default=False
        If True, also return per-moment contributions

    Returns
    --------
    total_metric : float
        Weighted, normalized metric over all patches and moments
    per_moment_dict : dict, optional
        Contribution of each moment (if return_per_moment=True)
    
    Examples
    --------
    >>> import torch
    >>> from FlexCNN_for_Medical_Physics.custom_criteria import patchwise_moment_metric
    >>> 
    >>> pred = torch.randn(4, 1, 128, 128)
    >>> target = torch.randn(4, 1, 128, 128)
    >>> 
    >>> # Simple usage with defaults
    >>> metric = patchwise_moment_metric(pred, target)
    >>> 
    >>> # With per-moment breakdown
    >>> metric, per_moment = patchwise_moment_metric(pred, target, return_per_moment=True)
    >>> print(f"Total: {metric:.4f}, Per-moment: {per_moment}")
    
    Notes
    -----
    - Defaults match PatchwiseMomentLoss for consistency
    - If use_poisson_normalization=True, only moments 1 and 2 are supported
    """
    if use_poisson_normalization and any(moment > 2 for moment in moments):
        raise ValueError(
            "Poisson normalization only supports moments [1, 2]. "
            "Disable use_poisson_normalization for higher moments."
        )

    # Call shared core computation with configured normalization
    total_metric, per_moment_dict = _compute_patchwise_moments(
        pred=batch_pred,
        target=batch_target,
        moments=moments,
        moment_weights=moment_weights if moment_weights is not None else {k: 1.0 for k in moments},
        patch_size=patch_size,
        stride=stride,
        eps=eps,
        patch_weighting=patch_weighting,
        patch_weight_min=patch_weight_min,
        patch_weight_max=patch_weight_max,
        max_patch_masked=max_patch_masked,
        use_poisson_normalization=use_poisson_normalization,
        scale=scale
    )
    
    # Convert tensor to scalar for metric output
    total_metric_scalar = total_metric.cpu().item()
    
    if return_per_moment:
        return total_metric_scalar, per_moment_dict
    else:
        return total_metric_scalar


def custom_metric(batch_A, batch_B):
    """
    Simple wrapper for patchwise_moment_metric with default parameters.
    
    Convenience function that provides a simple interface for evaluation.
    Uses all default parameters from defaults.py.
    
    Parameters
    ----------
    batch_A : torch.Tensor
        Predicted images, shape [B, C, H, W]
    batch_B : torch.Tensor
        Target images, same shape as batch_A
    
    Returns
    -------
    float
        Patchwise moment metric value
    
    Examples
    --------
    >>> from FlexCNN_for_Medical_Physics.custom_criteria import custom_metric
    >>> metric = custom_metric(pred_batch, target_batch)
    """
    return patchwise_moment_metric(batch_A, batch_B)
