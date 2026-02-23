
"""
Core patchwise moment computation for statistical matching.
All error/penalty logic is centralized and minimal.
Metrics return NaN for pathological cases; losses always return a differentiable penalty.
"""

import torch

def _compute_patchwise_moments(
    pred,
    target,
    moments,
    moment_weights,
    patch_size,
    stride,
    eps,
    patch_weighting,
    patch_weight_min,
    patch_weight_max,
    max_patch_masked,
    use_poisson_normalization,
    scale,
    counts_per_bq,
    pathological_penalty,
    return_penalty_for_pathological
):
    """
    Core patchwise moment computation for both training (loss) and evaluation (metric).
    Metrics return NaN for pathological cases; losses always return a differentiable penalty.
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
    if moment_weights is None:
        moment_weights = {k: 1.0 for k in moments}

    def _return_penalty():
        penalty_dict = {k: pathological_penalty for k in moments}
        if return_penalty_for_pathological:
            penalty_tensor = torch.tensor(pathological_penalty, device=pred.device, dtype=pred.dtype)
            return penalty_tensor, penalty_dict
        else:
            penalty_tensor = pred.mean() * 0.0 + pathological_penalty
            return penalty_tensor, penalty_dict

    # Pathological check: NaN or negative mean in pred
    if torch.isnan(pred).any() or pred.mean().item() < 0:
        return _return_penalty()

    B, C, H, W = pred.shape
    p, s = patch_size, stride
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

    pred_patches = pred.unfold(2, p, s).unfold(3, p, s)
    target_patches = target.unfold(2, p, s).unfold(3, p, s)
    num_patches = num_patches_h * num_patches_w
    pred_patches = pred_patches.contiguous().view(B, C, num_patches, -1)
    target_patches = target_patches.contiguous().view(B, C, num_patches, -1)

    if use_poisson_normalization:
        pred_patches = pred_patches * counts_per_bq
        target_patches = target_patches * counts_per_bq

    patch_mean = target_patches.mean(dim=-1)
    patch_min = patch_mean.min(dim=-1, keepdim=True)[0]
    patch_max = patch_mean.max(dim=-1, keepdim=True)[0]
    if patch_weighting == 'scaled':
        patch_weights = patch_weight_min + (patch_mean - patch_min) / (patch_max - patch_min + eps) * (patch_weight_max - patch_weight_min)
    elif patch_weighting == 'energy':
        patch_energy = (target_patches ** 2).mean(dim=-1)
        patch_weights = patch_energy / (patch_energy.sum(dim=-1, keepdim=True) + eps)
    elif patch_weighting == 'mean':
        patch_weights = patch_mean / (patch_mean.sum(dim=-1, keepdim=True) + eps)
    else:
        patch_weights = torch.ones_like(patch_mean)

    patch_mask = (patch_mean > max_patch_masked).float()
    patch_weights = patch_weights * patch_mask

    # All patches masked: pathological
    if patch_weights.sum() < eps:
        return _return_penalty()

    total_loss = 0.0
    per_moment_dict = {}
    for k in moments:
        if use_poisson_normalization:
            if k == 1:
                target_m = patch_mean
                pred_m = pred_patches.mean(dim=-1)
                denom = target_m + eps
            elif k == 2:
                target_var = (target_patches - patch_mean.unsqueeze(-1)) ** 2
                pred_var = (pred_patches - pred_patches.mean(dim=-1, keepdim=True)) ** 2
                target_m = torch.sqrt(target_var.mean(dim=-1) + eps)
                pred_m = torch.sqrt(pred_var.mean(dim=-1) + eps)
                denom = torch.sqrt(torch.clamp(patch_mean, min=0.0) + eps)
            else:
                raise RuntimeError(f"Unexpected moment {k} in Poisson mode")
        else:
            if k == 1:
                target_m = patch_mean
                pred_m = pred_patches.mean(dim=-1)
                denom = torch.ones_like(target_m)
            else:
                pred_c = pred_patches - pred_patches.mean(dim=-1, keepdim=True)
                target_c = target_patches - patch_mean.unsqueeze(-1)
                pred_m = (pred_c ** k).mean(dim=-1)
                target_m = (target_c ** k).mean(dim=-1)
                if scale == 'std':
                    sigma = torch.sqrt((target_c ** 2).mean(dim=-1) + eps)
                    denom = sigma ** k + eps
                elif scale == 'mean':
                    denom = (patch_mean ** k) + eps
                else:
                    denom = torch.ones_like(target_m)
        rel_diff = torch.abs(pred_m - target_m) / (denom + eps)
        weighted_patch_diff = (rel_diff * patch_weights).sum(dim=-1).mean(dim=[0, 1])
        weighted_moment_diff = weighted_patch_diff * moment_weights.get(k, 1.0)
        total_loss += weighted_moment_diff
        per_moment_dict[k] = weighted_moment_diff.detach().cpu().item()
    total_loss = total_loss / sum(moment_weights.get(k, 1.0) for k in moments)
    return total_loss, per_moment_dict
