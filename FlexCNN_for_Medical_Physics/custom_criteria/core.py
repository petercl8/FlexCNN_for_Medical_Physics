"""
Core patchwise moment computation for statistical matching.

This module contains the shared computational engine used by both training losses
and evaluation metrics. All parameters are required (no defaults) to ensure explicit
configuration by callers.
"""

import torch


def _compute_patchwise_moments(
    pred: torch.Tensor,
    target: torch.Tensor,
    moments: list,
    moment_weights: dict | None,
    patch_size: int,
    stride: int,
    eps: float,
    patch_weighting: str,
    patch_weight_min: float,
    patch_weight_max: float,
    max_patch_masked: float,
    use_poisson_normalization: bool,
    scale: str,
    counts_per_bq: float
):
    """
    Core patchwise moment computation for both training (loss) and evaluation (metric).
    
    This function extracts overlapping patches from predicted and target images, computes
    statistical moments for each patch, and returns the weighted difference. Supports both 
    physics-informed Poisson normalization for PET reconstruction and generic moment matching 
    for arbitrary moment orders.
    
    **All parameters are required** - no defaults provided. Callers (loss/metric functions) 
    must explicitly provide all configuration values.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted batch, shape [B, C, H, W]
    target : torch.Tensor
        Target batch, same shape as pred
    moments : list of int
        Which moments to compute. 1=mean, 2=variance/std, 3=skewness, 4=kurtosis, etc.
    moment_weights : dict or None
        Relative importance of each moment. Keys must match values in moments list.
        If None, all moments weighted equally (1.0 for each).
    patch_size : int
        Side length of square patches to extract (in pixels).
    stride : int
        Stride for patch extraction. Smaller values create more overlapping patches.
    eps : float
        Small constant for numerical stability in divisions.
    patch_weighting : str
        How to weight patches based on activity level:
        - 'scaled': Linear scaling between patch_weight_min and patch_weight_max
        - 'energy': Weight by patch energy (L2 norm squared)
        - 'mean': Weight by patch mean activity
        - 'none': Uniform weighting (all patches equal)
    patch_weight_min : float
        Minimum weight when using 'scaled' weighting mode.
    patch_weight_max : float
        Maximum weight when using 'scaled' weighting mode.
    max_patch_masked : float
        Mask (ignore) patches with mean activity ≤ this threshold.
    use_poisson_normalization : bool
        If True, uses physics-informed Poisson normalization (PET-specific):
        - Moment 1 (mean): normalized by patch mean (fractional bias)
        - Moment 2 (std): normalized by sqrt(patch mean) (Poisson noise scale)
        - Requires moments to contain only values ≤ 2.
        If False, uses generic normalization by mean^k or std^k based on 'scale' parameter.
    scale : str
        Normalization method when use_poisson_normalization=False.
        - 'mean': normalize moment k by mean^k
        - 'std': normalize moment k by std^k
    counts_per_bq : float
        Counts-per-activity scale used to convert patches to counts when
        use_poisson_normalization=True.
    
    Returns
    -------
    total_loss : torch.Tensor
        Scalar loss tensor (differentiable for backprop).
    per_moment_dict : dict
        Dictionary mapping moment order to its contribution: {1: value, 2: value, ...}
        Values are already detached and converted to Python floats.
    
    Raises
    ------
    ValueError
        If use_poisson_normalization=True but moments contains values > 2.
        If moment_weights keys don't match values in moments list.
        If patch_size is larger than image dimensions.
    
    Notes
    -----
    - This function is the single source of truth for patchwise moment computation.
    - Returns tensors for training (differentiable) and per-moment dict for analysis.
    - Patch weighting and masking are applied before aggregating moment differences.
    - PET mode is recommended for photon-counting imaging (PET, SPECT, CT).
    - Generic mode supports arbitrary moment orders for general texture matching.
    """
    # Validation
    if use_poisson_normalization and max(moments) > 2:
        raise ValueError(
            f"Poisson normalization mode requires moments ≤ 2, but got moments={moments}. "
            f"Either set use_poisson_normalization=False or use only moments [1, 2]."
        )
    
    if moment_weights is not None:
        invalid_keys = set(moment_weights.keys()) - set(moments)
        if invalid_keys:
            raise ValueError(
                f"moment_weights contains keys {invalid_keys} not present in moments={moments}"
            )
    
    # Default weights: all moments weighted equally
    if moment_weights is None:
        moment_weights = {k: 1.0 for k in moments}
    
    B, C, H, W = pred.shape
    p, s = patch_size, stride
    
    # -------------------
    # Crop to full patches only
    # -------------------
    num_patches_h = (H - p) // s + 1
    num_patches_w = (W - p) // s + 1
    if num_patches_h <= 0 or num_patches_w <= 0:
        raise ValueError(
            f"Patch size ({patch_size}) larger than image dimensions ({H}x{W}). "
            f"Reduce patch_size or check input shape."
        )
    max_h = s * (num_patches_h - 1) + p
    max_w = s * (num_patches_w - 1) + p
    pred = pred[:, :, :max_h, :max_w]
    target = target[:, :, :max_h, :max_w]
    
    # -------------------
    # Extract patches
    # Shape: [B, C, num_patches, patch_size^2]
    # -------------------
    pred_patches = pred.unfold(2, p, s).unfold(3, p, s)
    target_patches = target.unfold(2, p, s).unfold(3, p, s)
    num_patches = num_patches_h * num_patches_w
    pred_patches = pred_patches.contiguous().view(B, C, num_patches, -1)
    target_patches = target_patches.contiguous().view(B, C, num_patches, -1)
    
    # Convert to counts for Poisson normalization
    if use_poisson_normalization:
        pred_patches = pred_patches * counts_per_bq
        target_patches = target_patches * counts_per_bq

    # -------------------
    # Compute patch mean (needed for weighting and masking)
    # -------------------
    patch_mean = target_patches.mean(dim=-1)  # [B, C, num_patches]
    
    # -------------------
    # Compute patch weights (importance)
    # -------------------
    patch_min = patch_mean.min(dim=-1, keepdim=True)[0]
    patch_max = patch_mean.max(dim=-1, keepdim=True)[0]
    
    if patch_weighting == 'scaled':
        # Scale between patch_weight_min and patch_weight_max per image
        patch_weights = patch_weight_min + \
                        (patch_mean - patch_min) / (patch_max - patch_min + eps) * \
                        (patch_weight_max - patch_weight_min)
    elif patch_weighting == 'energy':
        patch_energy = (target_patches ** 2).mean(dim=-1)
        patch_weights = patch_energy / (patch_energy.sum(dim=-1, keepdim=True) + eps)
    elif patch_weighting == 'mean':
        patch_weights = patch_mean / (patch_mean.sum(dim=-1, keepdim=True) + eps)
    else:  # 'none' or any other value
        patch_weights = torch.ones_like(patch_mean)
    
    # -------------------
    # Mask low-activity patches
    # -------------------
    patch_mask = (patch_mean > max_patch_masked).float()
    patch_weights = patch_weights * patch_mask  # zero out masked patches
    
    # -------------------
    # Precompute centered deviations (needed for moment 2 and higher)
    # -------------------
    target_mean_centered = target_patches - patch_mean.unsqueeze(-1)
    pred_mean_centered = pred_patches - pred_patches.mean(dim=-1, keepdim=True)
    
    # -------------------
    # Moment computation
    # -------------------
    per_moment_dict = {}
    total_loss = 0.0
    
    for k in moments:
        if use_poisson_normalization:
            # PET-specific physics-informed normalization
            if k == 1:
                # Mean: normalize by patch mean (fractional bias)
                target_m = patch_mean
                pred_m = pred_patches.mean(dim=-1)
                denom = target_m + eps
            elif k == 2:
                # Std: normalize by sqrt(patch mean) (Poisson scaling)
                target_var = (target_mean_centered ** 2).mean(dim=-1)
                pred_var = (pred_mean_centered ** 2).mean(dim=-1)
                target_m = torch.sqrt(target_var + eps)
                pred_m = torch.sqrt(pred_var + eps)
                # Clamp patch_mean to prevent sqrt of negative values
                denom = torch.sqrt(torch.clamp(patch_mean, min=0.0) + eps)
            else:
                # Should never reach here due to validation
                raise RuntimeError(f"Unexpected moment {k} in Poisson mode")
        else:
            # Generic normalization for arbitrary moments
            if k == 1:
                # First moment: mean
                target_m = patch_mean
                pred_m = pred_patches.mean(dim=-1)
                denom = torch.ones_like(target_m)  # No normalization for mean
            else:
                # Higher moments: central moments
                pred_c = pred_patches - pred_patches.mean(dim=-1, keepdim=True)
                target_c = target_patches - patch_mean.unsqueeze(-1)
                pred_m = (pred_c ** k).mean(dim=-1)
                target_m = (target_c ** k).mean(dim=-1)
                
                if scale == 'std':
                    sigma = torch.sqrt((target_c**2).mean(dim=-1) + eps)
                    denom = sigma**k + eps
                elif scale == 'mean':
                    denom = (patch_mean**k) + eps
                else:
                    denom = torch.ones_like(target_m)
        
        # Relative difference per patch
        rel_diff = torch.abs(pred_m - target_m) / (denom + eps)
        
        # Aggregate weighted by patch importance
        weighted_patch_diff = (rel_diff * patch_weights).sum(dim=-1).mean(dim=[0, 1])
        
        # Apply moment weight/scale
        weighted_moment_diff = weighted_patch_diff * moment_weights.get(k, 1.0)
        total_loss += weighted_moment_diff
        per_moment_dict[k] = weighted_moment_diff.detach().cpu().item()
    
    # Normalize by sum of moment weights
    total_loss = total_loss / sum(moment_weights.get(k, 1.0) for k in moments)
    
    return total_loss, per_moment_dict
