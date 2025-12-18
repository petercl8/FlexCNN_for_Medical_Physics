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


import torch

def patchwise_moment_metric(batch_pred,
                            batch_target,
                            moments=[1,2,3],
                            moment_weights={1:0.8, 2:1.0, 3:0.001},   # dict, e.g., {2:1.0, 3:0.001}
                            patch_size=16,
                            stride=8,
                            eps=1e-6,
                            patch_weighting='mean',  # 'mean', 'energy', 'none'
                            return_per_moment=False):
    """
    Compute a patchwise moment metric with separate patch and moment weighting.

    Args:
        batch_pred (torch.Tensor): [B, C, H, W] predicted images
        batch_target (torch.Tensor): [B, C, H, W] target images
        moments (list of int): which central moments to compute, e.g., [2,3]
        moment_weights (dict or None): {moment: weight}, defaults to 1 for all moments
        patch_size (int): size of square patches
        stride (int): stride between patches
        eps (float): small number to avoid division by zero
        patch_weighting (str): 'mean', 'energy', or 'none'
        return_per_moment (bool): if True, return per-moment dict along with scalar

    Returns:
        total_metric (float): weighted mean metric across moments and patches
        per_moment_dict (dict): per-moment metric (CPU floats), optional
    """
    B, C, H, W = batch_pred.shape
    p = patch_size
    s = stride

    # Determine number of patches that fully fit
    num_patches_h = (H - p) // s + 1
    num_patches_w = (W - p) // s + 1

    if num_patches_h <= 0 or num_patches_w <= 0:
        raise ValueError("Patch size larger than image dimensions.")

    # Slice to only full patches
    max_h = s * (num_patches_h - 1) + p
    max_w = s * (num_patches_w - 1) + p
    batch_pred = batch_pred[:, :, :max_h, :max_w]
    batch_target = batch_target[:, :, :max_h, :max_w]

    # Extract patches: [B, C, num_patches, patch_size*patch_size]
    pred_patches = batch_pred.unfold(2, p, s).unfold(3, p, s)
    target_patches = batch_target.unfold(2, p, s).unfold(3, p, s)
    num_patches = num_patches_h * num_patches_w

    pred_patches = pred_patches.contiguous().view(B, C, num_patches, -1)
    target_patches = target_patches.contiguous().view(B, C, num_patches, -1)

    # -------------------
    # Compute patch weights
    # -------------------
    if patch_weighting == 'mean':
        patch_means = target_patches.mean(dim=-1)  # [B, C, num_patches]
        image_means = batch_target.mean(dim=[2,3], keepdim=True) + eps  # [B, C, 1, 1]
        patch_weights = (patch_means / image_means).clamp(min=0.0)
    elif patch_weighting == 'energy':
        patch_energy = (target_patches**2).mean(dim=-1)
        total_energy = patch_energy.sum(dim=-1, keepdim=True) + eps
        patch_weights = patch_energy / total_energy
    else:
        patch_weights = torch.ones_like(target_patches.mean(dim=-1))

    patch_weights = patch_weights / (patch_weights.sum(dim=-1, keepdim=True) + eps)  # normalize per image/channel

    # -------------------
    # Moment-level metric computation
    # -------------------
    per_moment_dict = {}
    total_metric = 0.0

    # Default moment weights = 1 if not provided
    if moment_weights is None:
        moment_weights = {k: 1.0 for k in moments}

    for k in moments:
        # Compute central moments
        target_mean = target_patches.mean(dim=-1, keepdim=True)
        pred_mean = pred_patches.mean(dim=-1, keepdim=True)

        target_c = target_patches - target_mean
        pred_c = pred_patches - pred_mean

        target_m = (target_c**k).mean(dim=-1)  # [B, C, num_patches]
        pred_m = (pred_c**k).mean(dim=-1)

        # Relative difference
        rel_diff = torch.abs(pred_m - target_m) / (torch.abs(target_m) + eps)

        # Weighted mean across patches (patch_weights), then batch/channels
        weighted_patch_diff = (rel_diff * patch_weights).sum(dim=-1).mean(dim=[0,1])

        # Apply moment weight
        weighted_moment_diff = weighted_patch_diff * moment_weights.get(k, 1.0)

        total_metric += weighted_moment_diff
        per_moment_dict[k] = weighted_moment_diff.cpu().item()

    # Normalize by sum of moment weights
    total_metric = total_metric / sum(moment_weights.get(k, 1.0) for k in moments)

    if return_per_moment:
        return total_metric.cpu().item(), per_moment_dict
    else:
        return total_metric.cpu().item()


# Wrap in a function for a simple interface
def custom_metric(batch_A, batch_B):
    return patchwise_moment_metric(batch_A,
                            batch_B,
                            moments=[1,2,3],
                            moment_weights={1:0.8, 2:1.0, 3:0.001},   # dict, e.g., {2:1.0, 3:0.001}
                            patch_size=16,
                            stride=8,
                            eps=1e-6,
                            patch_weighting='mean',  # 'mean', 'energy', 'none'
                            return_per_moment=False)


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