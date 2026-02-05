# -*- coding: utf-8 -*-
"""
User Parameters for FlexCNN Medical Physics Pipeline

This module contains all user-configurable parameters for tuning, training, testing,
and visualizing CNNs for PET image reconstruction.

To use in a script:
    from user_params import get_params
    params = get_params()

To use in a notebook with overrides:
    import user_params
    params = user_params.get_params()
    # Override specific parameters
    params['run_mode'] = 'train'
    params['network_type'] = 'ATTEN'
"""

#####################
### General Setup ###
#####################

## Basic Options ##
run_mode = 'train'  # Options: 'tune', 'train', 'test', 'visualize', 'none'
network_type = 'ACT'  # 'ACT', 'ATTEN', 'CONCAT', 'FROZEN_COFLOW', 'FROZEN_COUNTERFLOW'
train_SI = True  # If working with GAN or SUP networks, True for Sinogram-->Image, False for Image-->Sinogram
use_cache = False  # Cache dataset to Google Colab VM?
cache_max_gb = 40  # Max size for cache
cache_dir = '/content/cache'

plot_mode = 'inline'  # Options: 'always', 'inline', 'never'

## Network Input/Output Sizes ##
gen_sino_size = 288  # Options: 180, 288, 320. Resize input sinograms to this size
gen_image_size = 180  # Image size (Options: 90)
gen_sino_channels = 3  # Number of sinogram channels for network currently being trained
gen_image_channels = 1  # Number of image channels for network currently being trained

SI_normalize = False  # For sino-->image mappings: normalize CNN outputs
IS_normalize = False  # For image-->sinogram mappings: normalize CNN outputs

## Scales ##
act_recon1_scale = 3.350  # Scale factor for optional recon1
act_recon2_scale = 1.998  # Scale factor for optional recon2
act_sino_scale = 0.342  # Scale factor for sinograms (if not normalizing)
act_image_scale = 1  # Scale factor for images (if not normalizing)
atten_image_scale = 308.335  # Scale factor for attenuation images
atten_sino_scale = 39.187140258382726  # Scale factor for attenuation sinograms

## Resources ##
num_CPUs = 12  # T4:8 L4/V100: 12
num_GPUs = 1
CPUs_per_trial = 2  # Resources per Ray Tune trial
GPUs_per_trial = 1

device_opt = 'cuda'  # Options: 'sense', 'cuda', 'cpu'

skip_local_package_installs = True  # Local only: skip package checks after first run
ray_tune_version = None  # Optional: Pin Ray Tune version (e.g., '2.9.0')
skip_colab_git_update = False  # Colab only: Skip git pull when setting up repository
setup_mode_type = 'walk'  # Local only: 'walk' (reload modules) or 'install' (editable pip)
base_repo_path = None  # Optional: override repo root detection

## Github Repository ##
github_username = 'petercl8'
repo_name = 'FlexCNN_for_Medical_Physics'

## Directories ##
project_colab_dirPath = '/content/drive/MyDrive/Colab/Working/'
project_local_dirPath = r'C:\Users\Peter Lindstrom\My Drive (lindstrom.peter@gmail.com)\Colab\Working'
local_repo_dirPath = r'C:\FlexCNN_cloned'

data_dirName = 'dataset-sets'
plot_dirName = 'plots'
checkpoint_dirName = 'checkpoints'
num_examples = -1  # Number of examples to load (-1 = all)

############
## Tuning ##
############

tune_csv_file = 'frame-ACT-180-padZeros-tunedSSIM-B'
tune_exp_name = 'search-ACT-180-padZeros-tunedSSIM-Bs'
tune_scheduler = 'ASHA'  # 'FIFO' or 'ASHA'
tune_dataframe_fraction = 0.33
tune_restore = False
tune_minutes = 4*60
tune_metric = 'SSIM'  # 'MSE', 'SSIM', 'CUSTOM', or QA metrics
tune_even_reporting = True
tune_batches_per_report = 10
tune_examples_per_report = 4*256
tune_grace_period = 4
tune_max_t = 48
tune_report_for = 'val'  # 'val' or 'qa'
tune_eval_batch_size = 64
tune_augment = ('SI', True)
tune_debug = False
tune_force_fixed_config = False
tune_search_alg = 'optuna'  # 'optuna' or 'hyperopt'

## Training Files ##
tune_act_sino_file = 'train-highCountSino-320x257.npy'
tune_act_image_file = 'train-actMap.npy'
tune_atten_sino_file = None
tune_atten_image_file = None
tune_act_recon1_file = None
tune_act_recon2_file = None

## Cross Validation Set ##
tune_val_act_sino_file = 'val-highCountSino-320x257.npy'
tune_val_act_image_file = 'val-actMap.npy'
tune_val_atten_image_file = None
tune_val_atten_sino_file = None

## QA Phantoms ##
tune_qa_hot_weight = 0.5
tune_qa_act_sino_file = None
tune_qa_act_image_file = None
tune_qa_atten_image_file = None
tune_qa_atten_sino_file = None
tune_qa_hotMask_file = None
tune_qa_hotBackgroundMask_file = None
tune_qa_coldMask_file = None
tune_qa_coldBackgroundMask_file = None

## Unlikely to Change ##
tune_storage_dirName = 'searches'
tune_dataframe_dirName = 'dataframes-tune'

##############
## Training ##
##############

train_checkpoint_file = 'temp'
train_load_state = False
train_save_state = False
train_epochs = 200
train_display_step = 10
train_sample_division = 1
train_show_times = True

## Data Files & Augmentations ##
train_shuffle = True
train_augment = ('SI', True)
train_act_sino_file = 'train-highCountSino-320x257.npy'
train_act_image_file = 'train-actMap.npy'
train_atten_image_file = 'train-attenMap.npy'
train_atten_sino_file = 'train-attenSino-382x513.npy'
train_act_recon1_file = None
train_act_recon2_file = None

###########
# Testing #
###########

test_dataframe_dirName = 'TestOnFull'
test_csv_file = 'combined-tunedLowSSIM-trainedLowSSIM-onTestSet-wMLEM'
test_checkpoint_file = 'checkpoint-tunedLowSSIM-trainedLowSSIM-100epochs'
test_display_step = 15
test_batch_size = 25
test_chunk_size = 875
testset_size = 35000
test_begin_at = 0
test_compute_MLEM = False
test_merge_dataframes = True
test_show_times = False
test_shuffle = False
test_sample_division = 1

## Select Data Files ##
test_act_sino_file = 'test-highCountSino-180x180.npy'
test_act_image_file = 'test-actMap.npy'
test_atten_image_file = None
test_atten_sino_file = None
test_act_recon1_file = 'test-highCountImage.npy'
test_act_recon2_file = 'test-obliqueImage.npy'

####################
## Visualize Data ##
####################

visualize_act_sino_file = 'train-highCountSino-382x513.npy'
visualize_act_image_file = 'train-actMap.npy'
visualize_act_recon1_file = 'train-highCountImage.npy'
visualize_act_recon2_file = 'train-obliqueImage.npy'
visualize_atten_image_file = None
visualize_atten_sino_file = None
visualize_checkpoint_file = 'checkpoint-new_data-old_net-180x180-temp'
visualize_batch_size = 10
visualize_offset = 0
visualize_shuffle = True


def get_params():
    """
    Convert all module-level variables to a dictionary.
    
    Returns:
        dict: Dictionary containing all user parameters
    """
    params = {}
    current_module = __import__(__name__)
    
    for name in dir(current_module):
        # Skip private attributes, functions, and imported modules
        if not name.startswith('_') and name != 'get_params':
            value = getattr(current_module, name)
            # Only include simple types (not functions, classes, or modules)
            if not callable(value) and not isinstance(value, type):
                params[name] = value
    
    return params