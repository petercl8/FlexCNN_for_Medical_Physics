import torch
import numpy as np
from skimage.metrics import structural_similarity
from .cropping import crop_image_tensor_with_corner

######################
## Metric Functions ##
######################

## Metrics which take only single images as inputs ##
## ----------------------------------------------- ##
def SSIM(image_A, image_B, win_size=-1):
    '''
    Function to return the SSIM for two 2D images.

    image_A:        pytorch tensor for a single image
    image_B:        pytorch tensor for a single image
    win_size:       window size to use when computing the SSIM. This must be an odd number. If =-1, the full size of the image is used (or full size-1 so it's odd).
    '''

    if win_size == -1:   # The default shape of the window size is the same size as the image.
        x = image_A.shape[2]
        win_size = (x if x % 2 == 1 else x-1) # Guarantees the window size is odd.

    image_A_npy = image_A.detach().squeeze().cpu().numpy()
    image_B_npy = image_B.detach().squeeze().cpu().numpy()

    max_value = max([np.amax(image_A_npy, axis=(0,1)), np.amax(image_B_npy, axis=(0,1))])   # Find maximum among the images
    min_value = min([np.amin(image_A_npy, axis=(0,1)), np.amin(image_B_npy, axis=(0,1))])   # Find minimum among the images
    data_range = max_value-min_value

    SSIM_image = structural_similarity(image_A_npy, image_B_npy, data_range=data_range, gaussian_weights=False, use_sample_covariance=False, win_size=win_size)

    return SSIM_image

## Metrics which take either batches or images as inputs ##
## ----------------------------------------------------- ##
def MSE(image_A, image_B):
    '''
    Function to return the mean square error for two 2D images (or two batches of images).

    image_A:        pytorch tensor for a single image
    image_B:        pytorch tensor for a single image
    '''
    image_A_npy = image_A.detach().squeeze().cpu().numpy()
    image_B_npy = image_B.detach().squeeze().cpu().numpy()

    return torch.mean((image_A-image_B)**2).item()

def NMSE(image_A, image_B):
    '''
    Function to return the normalized mean square error for two 2D images (or two batches of images).

    image_A:        pytorch tensor for a single image (reference image)
    image_B:        pytorch tensor for a single image
    '''
    image_A_npy = image_A.detach().squeeze().cpu().numpy()
    image_B_npy = image_B.detach().squeeze().cpu().numpy()

    return (torch.mean((image_A-image_B)**2)/torch.mean(image_A**2)).item()

def MAE(image_A, image_B):
    '''
    Function to return the mean absolute error for two 2D images (or two batches of images).

    image_A:        pytorch tensor for a single image
    image_B:        pytorch tensor for a single image
    '''
    image_A_npy = image_A.detach().squeeze().cpu().numpy()
    image_B_npy = image_B.detach().squeeze().cpu().numpy()

    return torch.mean(torch.abs(image_A-image_B)).item()

def patchwise_moment_metric(batch_pred,
                            batch_target,
                            moments=[1,2],                    # 1=mean, 2=std
                            moment_weights={1:2.0, 2:1.0},    # relative importance of each moment
                            patch_size=8,
                            stride=4,
                            eps=1e-6,
                            patch_weighting='scaled',          # 'scaled', 'energy', 'mean', 'none'
                            patch_weight_min=0.33,
                            patch_weight_max=1.0,
                            min_patch_mean=1e-3,               # threshold to ignore very low-activity patches
                            return_per_moment=False):
    """
    Patchwise moment metric for PET reconstructions (mean + std only), 
    with optional low-count patch masking and tunable patch/moment weighting.

    -----------------------------
    Physics-informed normalization:
    - Mean differences normalized by patch mean (fractional bias)
    - Std differences normalized by sqrt(patch mean) (Poisson noise scale)
    -----------------------------
    
    Parameters:
    -----------
    batch_pred : torch.Tensor
        Predicted batch, shape [B, C, H, W]
    batch_target : torch.Tensor
        Target batch, same shape as batch_pred
    moments : list
        Which moments to compute: 1=mean, 2=std
    moment_weights : dict
        Relative importance of each moment in final metric
    patch_size : int
        Size of square patch
    stride : int
        Stride between patches
    eps : float
        Small constant for numerical stability
    patch_weighting : str
        How to weight patches: 'scaled', 'energy', 'mean', or 'none'
    patch_weight_min : float
        Minimum weight when using 'scaled' weighting
    patch_weight_max : float
        Maximum weight when using 'scaled' weighting
    min_patch_mean : float
        Ignore patches with mean below this threshold (stability)
    return_per_moment : bool
        If True, also return per-moment contributions

    Returns:
    --------
    total_metric : float
        Weighted, normalized metric over all patches and moments
    per_moment_dict : dict, optional
        Contribution of each moment (if return_per_moment=True)
    """

    B, C, H, W = batch_pred.shape
    p, s = patch_size, stride

    # -------------------
    # Crop to full patches only
    # -------------------
    num_patches_h = (H - p) // s + 1
    num_patches_w = (W - p) // s + 1
    if num_patches_h <= 0 or num_patches_w <= 0:
        raise ValueError("Patch size larger than image dimensions.")
    max_h = s * (num_patches_h - 1) + p
    max_w = s * (num_patches_w - 1) + p
    batch_pred = batch_pred[:, :, :max_h, :max_w]
    batch_target = batch_target[:, :, :max_h, :max_w]

    # -------------------
    # Extract patches
    # Shape: [B, C, num_patches, patch_size^2]
    # -------------------
    pred_patches = batch_pred.unfold(2, p, s).unfold(3, p, s)
    target_patches = batch_target.unfold(2, p, s).unfold(3, p, s)
    num_patches = num_patches_h * num_patches_w
    pred_patches = pred_patches.contiguous().view(B, C, num_patches, -1)
    target_patches = target_patches.contiguous().view(B, C, num_patches, -1)

    # -------------------
    # Compute patch mean once
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
    else:
        patch_weights = torch.ones_like(patch_mean)

    # -------------------
    # Mask very low-activity patches
    # -------------------
    patch_mask = (patch_mean >= min_patch_mean).float()
    patch_weights = patch_weights * patch_mask  # zero out ignored patches

    # -------------------
    # Precompute centered deviations for std (moment 2)
    # -------------------
    target_mean_centered = target_patches - patch_mean.unsqueeze(-1)
    pred_mean_centered = pred_patches - pred_patches.mean(dim=-1, keepdim=True)

    # -------------------
    # Moment computation
    # -------------------
    per_moment_dict = {}
    total_metric = 0.0
    if moment_weights is None:
        moment_weights = {k:1.0 for k in moments}

    for k in moments:
        if k == 1:
            # Mean: normalize by patch mean
            target_m = patch_mean
            pred_m = pred_patches.mean(dim=-1)
            denom = target_m + eps
        elif k == 2:
            # Std: normalize by sqrt(patch mean) (Poisson scaling)
            target_var = (target_mean_centered ** 2).mean(dim=-1)
            pred_var = (pred_mean_centered ** 2).mean(dim=-1)
            target_m = torch.sqrt(target_var + eps)
            pred_m = torch.sqrt(pred_var + eps)
            denom = torch.sqrt(patch_mean + eps)
        else:
            raise ValueError("Only mean (1) and std (2) supported.")

        # Relative difference per patch
        rel_diff = torch.abs(pred_m - target_m) / denom

        # Aggregate weighted by patch importance
        weighted_patch_diff = (rel_diff * patch_weights).sum(dim=-1).mean(dim=[0,1])

        # Apply moment weight/scale
        weighted_moment_diff = weighted_patch_diff * moment_weights.get(k, 1.0)
        total_metric += weighted_moment_diff
        per_moment_dict[k] = weighted_moment_diff.cpu().item()

    # Normalize by sum of moment weights
    total_metric = total_metric / sum(moment_weights.get(k,1.0) for k in moments)

    if return_per_moment:
        return total_metric.cpu().item(), per_moment_dict
    else:
        return total_metric.cpu().item()



# Wrap in a function for a simple interface
def custom_metric(batch_A, batch_B):
    return patchwise_moment_metric(batch_A, batch_B)


###############################################
## Average or a Batch Metrics: Good for GANs ##
###############################################

# Range #
def range_metric(real, fake):
    '''
    Computes a simple metric which penalizes "fake" images in a batch for having a range different than the "real" images in a batch.
    Only a single metric number is returned.
    '''
    range_real = torch.max(real).item()-torch.min(real).item()
    range_fake = torch.max(fake).item()-torch.min(fake).item()

    return abs(range_real-range_fake)/(range_real+.1)

# Average #
def avg_metric(real, fake):
    '''
    Computes a simple metric which penalizes "fake" images in a batch for having an average value different than the "real" images in a batch.
    Only a single metric number is returned.
    '''
    avg_metric = abs((torch.mean(real).item()-torch.mean(fake).item())/(torch.mean(real)+.1).item())
    return avg_metric

# Pixel Variation #
def pixel_dist_metric(real, fake):
    '''
    Computes a metric which penalizes "fake" images for having a pixel distance different than the "real" images.

    real: real image tensor
    fake: fake image tensor
    '''
    def pixel_dist(image_tensor):
        '''
        Function for computing the pixel distance (standard deviation from mean) for a batch of images.
        For simplicity, it only looks at the 0th channel.
        '''
        array = image_tensor[:,0,:,:].detach().cpu().numpy().squeeze()
        sd = np.std(array, axis=0)
        avg=np.mean(sd)
        return(avg)

    pix_dist_fake = pixel_dist(fake)
    pix_dist_real = pixel_dist(real)

    return abs((pix_dist_real-pix_dist_fake)/(pix_dist_real+.1)) # The +0.1 in the denominators guarantees we don't divide by zero

###################
## Old Functions ##
###################


def LDM_OLD(real, fake, crop_size = 10, stride=10):
    '''
    Function to return the local distributions metric for two images.

    image_A:        pytorch tensor for a single image
    image_B:        pytorch tensor for a single image
    '''
    image_size = real.shape[2]

    i_max = int((image_size)/stride) # Maximum number of windows occurs when the stride equals the crop_size
    while (i_max-1)*stride + crop_size > image_size: # If stride < crop_size, we need fewer need to solve for the number of crops
        i_max += -1

    def crop_image_tensor_with_corner(A, corner=(0,0), crop_size=1):
        '''
        Function which returns a small, cropped version of an image.

        A           a batch of images with dimensions: (num_images, channel, height, width)
        corner      upper-left corner of window
        crop_size   size of croppiong window
        '''
        x_min = corner[1]
        x_max = corner[1]+crop_size
        y_min = corner[0]
        y_max = corner[0]+crop_size
        return A[:,:, y_min:y_max , x_min:x_max ]

    running_dist_score = 0
    running_avg_score = 0

    for i in range(0, i_max):
        for j in range(0, j_max):
            corner = (i*crop_size, j*crop_size)
            win_real = crop_image_tensor_with_corner(real, corner, crop_size)
            win_fake = crop_image_tensor_with_corner(fake, corner, crop_size)

            #range_score = range_metric(win_real, win_fake)
            avg_score = avg_metric(win_real, win_fake)
            pixel_dist_score = pixel_dist_metric(win_real, win_fake)

            running_dist_score += pixel_dist_score
            running_avg_score += avg_score

    combined_score = running_dist_score + running_avg_score

    return combined_score