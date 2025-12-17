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


def custom_patchwise_metric_detailed(batch_A, batch_B,
                                     patch_size=5,
                                     stride=2,
                                     max_moment=3,
                                     scale='mean',
                                     weights=None,
                                     return_per_moment=False):
    """
    Compute a patchwise moment metric between two image batches.
    
    Args:
        batch_A (torch.Tensor): predicted images, shape [B,C,H,W]
        batch_B (torch.Tensor): target images, same shape
        patch_size (int): size of patches
        stride (int): stride between patches
        max_moment (int): number of moments to compare
        scale (str): 'mean' or 'std' for higher moment scaling
        weights (list or None): optional weights for each moment
        return_per_moment (bool): if True, return a dict of per-moment scores
    
    Returns:
        float or dict: mean patchwise moment difference across all moments (default),
                       or dict of per-moment scores if return_per_moment=True
    """
    if weights is None:
        weights = [1.0] * max_moment

    B, C, H, W = batch_A.shape
    p = patch_size
    s = stride

    pred_patches = batch_A.unfold(2, p, s).unfold(3, p, s).contiguous().view(B, C, -1, p*p)
    target_patches = batch_B.unfold(2, p, s).unfold(3, p, s).contiguous().view(B, C, -1, p*p)

    per_moment_scores = {}
    total_score = 0.0

    for k in range(1, max_moment+1):
        target_mean = target_patches.mean(dim=-1, keepdim=True)
        pred_mean = pred_patches.mean(dim=-1, keepdim=True)

        if k == 1:
            moment_diff = torch.abs(pred_mean - target_mean).mean(dim=-1)
        else:
            pred_c = pred_patches - pred_mean
            target_c = target_patches - target_mean
            pred_m = (pred_c ** k).mean(dim=-1)
            target_m = (target_c ** k).mean(dim=-1)

            if scale == 'std':
                sigma = torch.sqrt((target_c**2).mean(dim=-1) + 1e-6)
                pred_m = pred_m / (sigma**k + 1e-6)
                target_m = target_m / (sigma**k + 1e-6)
            elif scale == 'mean':
                mean_val = target_mean.squeeze(-1) + 1e-6
                pred_m = pred_m / (mean_val**k)
                target_m = target_m / (mean_val**k)

            moment_diff = torch.abs(pred_m - target_m).mean(dim=-1)

        score = weights[k-1] * moment_diff.mean()
        per_moment_scores[k] = score.cpu().item()
        total_score += score

    if return_per_moment:
        return per_moment_scores
    else:
        return total_score.cpu().item()


# Wrap in a function for a simple interface
def custom_metric(batch_A, batch_B):
    return 0
    #return custom_patchwise_metric_detailed(batch_A, batch_B)


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