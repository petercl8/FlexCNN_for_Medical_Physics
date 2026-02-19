import torch
import pandas as pd
from .metrics import MSE, SSIM, custom_metric
from ..image_processing.reconstruction_projection import reconstruct
from ..image_processing.cropping import crop_image_tensor_by_factor


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


def reconstruct_images_and_update_test_dataframe(sino_tensor, CNN_output, act_map_scaled, test_dataframe, config, compute_MLEM=False, recon1=None, recon2=None):
    '''
    Function which: A) performs reconstructions (FBP and possibly ML-EM) if not provided
                    B) constructs a dataframe of metric values (MSE & SSIM) for these reconstructions, and also for the CNN output, with respect to the ground truth activity map.
                    C) concatenates this with the test dataframe passed to this function
                    D) returns the concatenated dataframe, mean metric values, and reconstructions

    sino_tensor:        sinogram tensor of shape [num, chan, height, width]
    CNN_output:         CNN reconstructions
    act_map_scaled:     ground truth activity map tensor
    test_dataframe:     dataframe to append metric values to
    config:             configuration dictionary containing gen_image_size, SI_normalize, SI_fixedScale
    compute_MLEM:       whether to compute ML-EM reconstructions (can be slow)
    recon1:             optional pre-computed reconstruction 1 tensor. If None, FBP is computed on-the-fly.
    recon2:             optional pre-computed reconstruction 2 tensor. If None, MLEM is computed on-the-fly.

    Note: MSE and SSIM are calculated using the metrics.py file, which are defined below in this module.
    '''

    # Construct or use pre-computed Reconstruction 1 #
    if recon1 is not None:
        recon1_output = recon1
    else:
        recon1_output = reconstruct(sino_tensor, config['gen_image_size'], config['SI_normalize'], config['SI_fixedScale'], recon_type='FBP')

    # Construct or use pre-computed Reconstruction 2 #
    if recon2 is not None:
        recon2_output = recon2
    else:
        if compute_MLEM==True:
            recon2_output = reconstruct(sino_tensor, config['gen_image_size'], config['SI_normalize'], config['SI_fixedScale'], recon_type='MLEM')
        else:
            recon2_output = recon1_output

    # Dataframes: build dataframes for every reconstruction technique/metric combination #
    batch_CNN_MSE,  mean_CNN_MSE   = calculate_metric(act_map_scaled, CNN_output, MSE,  return_dataframe=True, label='MSE (Network)')
    batch_CNN_SSIM,  mean_CNN_SSIM = calculate_metric(act_map_scaled, CNN_output, SSIM, return_dataframe=True, label='SSIM (Network)')
    batch_recon1_MSE,  mean_recon1_MSE   = calculate_metric(act_map_scaled, recon1_output, MSE,  return_dataframe=True, label='MSE (Recon1)')
    batch_recon1_SSIM,  mean_recon1_SSIM = calculate_metric(act_map_scaled, recon1_output, SSIM, return_dataframe=True, label='SSIM (Recon1)')
    batch_recon2_MSE, mean_recon2_MSE  = calculate_metric(act_map_scaled, recon2_output, MSE, return_dataframe=True, label='MSE (Recon2)')
    batch_recon2_SSIM, mean_recon2_SSIM= calculate_metric(act_map_scaled, recon2_output, SSIM,return_dataframe=True, label='SSIM (Recon2)')

    # Concatenate batch dataframes and larger running test dataframe
    add_frame = pd.concat([batch_CNN_MSE, batch_recon1_MSE, batch_recon2_MSE, batch_CNN_SSIM, batch_recon1_SSIM, batch_recon2_SSIM], axis=1)
    test_dataframe = pd.concat([test_dataframe, add_frame], axis=0)

    # Return dataframe, mean metrics, and reconstructions (generic names)
    return test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM, recon1_output, recon2_output


def update_tune_dataframe(tune_dataframe, tune_dataframe_path, model, config, metrics):
    '''
    Function to update the tune_dataframe for each trial run that makes it partway through the tuning process.

    tune_dataframe      a dataframe that stores model and IQA metric information for a particular trial
    model               model being trained (in tuning)
    config              configuration dictionary
    metrics             dictionary of metrics returned by evaluate_val() or evaluate_qa()

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

    # Build hyperparameters dict
    hyperparams = {
        'SI_dropout': SI_dropout,
        'SI_exp_kernel': SI_exp_kernel,
        'SI_gen_fill': SI_gen_fill,
        'SI_gen_hidden_dim': SI_gen_hidden_dim,
        'SI_gen_neck': SI_gen_neck,
        'SI_layer_norm': SI_layer_norm,
        'SI_normalize': SI_normalize,
        'SI_pad_mode': SI_pad_mode,
        'batch_size': batch_size,
        'gen_lr': gen_lr,
        'num_params': num_params
    }

    # Merge hyperparameters with metrics
    row_data = {**hyperparams, **metrics}

    # Concatenate Dataframe
    add_frame = pd.DataFrame(row_data, index=[0])
    tune_dataframe = pd.concat([tune_dataframe, add_frame], axis=0)

    # Save Dataframe to File
    tune_dataframe.to_csv(tune_dataframe_path, index=False)

    return tune_dataframe