import torch
import pandas as pd
from .metrics import MSE, SSIM, custom_metric
from .reconstruction_projection import reconstruct
from .cropping import crop_image_tensor_by_factor


##################################################
## Functions for Calculating Metrics Dataframes ##
##############################  ####################

## Calculate Arbitrary Metric ##
def calculate_metric(batch_A, batch_B, img_metric_function, return_dataframe=False, label='default', crop_factor=1):
    '''
    Function which calculates metric values for two batches of images.
    Returns either the average metric value for the batch or a dataframe of individual image metric values.

    batch_A:                tensor of images to compare [num, chan, height, width]
    batch_B:                tensor of images to compare [num, chan, height, width]
    img_metric_function:    a function which calculates a metric (MSE, SSIM, etc.) from two INDIVIDUAL images
    return_dataframe:       If False, then the average is returned.
                            Otherwise both the average, and a dataframe containing the metric values of the images in the batches, are returned.
    label:                  what to call dataframe, if it is created
    crop_factor:            factor by which to crop both batches of images. 1 = whole image is retained.
    '''
    
    import pandas as pd

    if crop_factor != 1:
        A = crop_image_tensor_by_factor(batch_A, crop_factor=crop_factor)
        B = crop_image_tensor_by_factor(batch_B, crop_factor=crop_factor)

    length = len(batch_A)
    metric_avg = 0
    metric_list = []

    for i in range(length):
        image_A = batch_A[i:i+1,:,:,:] # Using i:i+1 instead of just i preserves the dimensionality of the array
        image_B = batch_B[i:i+1,:,:,:]

        metric_value = img_metric_function(image_A, image_B)
        metric_avg += metric_value/length
        if return_dataframe==True:
            metric_list.append(metric_value)

    if return_dataframe==False:
        return metric_avg
    else:
        metric_frame = pd.DataFrame({label : metric_list})
        return metric_frame, metric_avg


def reconstruct_images_and_update_test_dataframe(sino_tensor, CNN_output, ground_image, test_dataframe, config, compute_MLEM=False):
    '''
    Function which: A) performs reconstructions (FBP and possibly ML-EM)
                    B) constructs a dataframe of metric values (MSE & SSIM) for these reconstructions, and also for the CNN output, with respect to the ground truth image.
                    C) concatenates this with the test dataframe passed to this function
                    D) returns the concatenated dataframe, mean metric values, and reconstructions

    sino_tensor:    sinogram tensor of shape [num, chan, height, width]
    CNN_output:     CNN reconstructions
    ground_image:   ground truth images
    test_dataframe: dataframe to append metric values to
    config:         configuration dictionary containing image_size, SI_normalize, SI_output_scale_fixed
    compute_MLEM:   whether to compute ML-EM reconstructions (can be slow)

    Note: MSE and SSIM are calculated using the metrics.py file, which are defined below in this module.
    '''

    # Construct Outputs #
    FBP_output = reconstruct(sino_tensor, config['image_size'], config['SI_normalize'], config['SI_output_scale_fixed'], recon_type='FBP')
    if compute_MLEM==True:
        MLEM_output = reconstruct(sino_tensor, config['image_size'], config['SI_normalize'], config['SI_output_scale_fixed'], recon_type='MLEM')
    else: # If not looking at ML-EM, don't waste time computing the MLEM images, which can take awhile.
        MLEM_output = FBP_output

    # Dataframes: build dataframes for every reconstruction technique/metric combination #
    batch_CNN_MSE,  mean_CNN_MSE   = calculate_metric(ground_image, CNN_output, MSE,  return_dataframe=True, label='MSE (Network)')
    batch_CNN_SSIM,  mean_CNN_SSIM = calculate_metric(ground_image, CNN_output, SSIM, return_dataframe=True, label='SSIM (Network)')
    batch_FBP_MSE,  mean_FBP_MSE   = calculate_metric(ground_image, FBP_output, MSE,  return_dataframe=True, label='MSE (FBP)')
    batch_FBP_SSIM,  mean_FBP_SSIM = calculate_metric(ground_image, FBP_output, SSIM, return_dataframe=True, label='SSIM (FBP)')
    batch_MLEM_MSE, mean_MLEM_MSE  = calculate_metric(ground_image, MLEM_output, MSE, return_dataframe=True, label='MSE (ML-EM)')
    batch_MLEM_SSIM, mean_MLEM_SSIM= calculate_metric(ground_image, MLEM_output, SSIM,return_dataframe=True, label='SSIM (ML-EM)')

    # Concatenate batch dataframes and larger running test dataframe
    add_frame = pd.concat([batch_CNN_MSE, batch_FBP_MSE, batch_MLEM_MSE, batch_CNN_SSIM, batch_FBP_SSIM, batch_MLEM_SSIM], axis=1)
    test_dataframe = pd.concat([test_dataframe, add_frame], axis=0)

    # Return a whole lot of stuff
    return test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_FBP_MSE, mean_FBP_SSIM, mean_MLEM_MSE, mean_MLEM_SSIM, FBP_output, MLEM_output


def update_tune_dataframe(tune_dataframe, tune_dataframe_path, model, config, mean_CNN_MSE, mean_CNN_SSIM, mean_CNN_CUSTOM):
    '''
    Function to update the tune_dataframe for each trial run that makes it partway through the tuning process.

    tune_dataframe      a dataframe that stores model and IQA metric information for a particular trial
    model               model being trained (in tuning)
    config              configuration dictionary
    mean_CNN_MSE        mean MSE for the CNN
    mean_CNN_SSIM       mean SSIM for the CNN
    mean_CNN_CUSTOM     mean custom metric for the CNN

    '''
    # Extract values from config dictionary
    SI_dropout =        config['SI_dropout']
    SI_exp_kernel =     config['SI_exp_kernel']
    SI_gen_fill =       config['SI_gen_fill']
    SI_gen_hidden_dim = config['SI_gen_hidden_dim']
    SI_gen_neck =       config['SI_gen_neck']
    SI_layer_norm =     config['SI_layer_norm']
    SI_normalize =      config['SI_normalize']
    SI_pad_mode =       config['SI_pad_mode']
    batch_size =        config['batch_size']
    gen_lr =            config['gen_lr']

    # Calculate number of trainable weights in CNN
    num_params = sum(map(torch.numel, model.parameters()))

    # Concatenate Dataframe
    add_frame = pd.DataFrame({'SI_dropout': SI_dropout, 'SI_exp_kernel': SI_exp_kernel, 'SI_gen_fill': SI_gen_fill, 'SI_gen_hidden_dim': SI_gen_hidden_dim,
                            'SI_gen_neck': SI_gen_neck, 'SI_layer_norm': SI_layer_norm, 'SI_normalize': SI_normalize, 'SI_pad_mode': SI_pad_mode, 'batch_size': batch_size,
                            'gen_lr': gen_lr, 'num_params': num_params, 'mean_CNN_MSE': mean_CNN_MSE, 'mean_CNN_SSIM': mean_CNN_SSIM, 'mean_CNN_CUSTOM': mean_CNN_CUSTOM}, index=[0])

    tune_dataframe = pd.concat([tune_dataframe, add_frame], axis=0)

    # Save Dataframe to File
    tune_dataframe.to_csv(tune_dataframe_path, index=False)

    return tune_dataframe