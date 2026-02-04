# -*- coding: utf-8 -*-
"""
This code can perform the following tasks:

*   Tune a CNN to directly reconstruct PET images from Sinograms (find a set of hyperparameters)
*   Train a network with a given set of hyperparameters
*   Test the network and record MSE and SSIM values for each image tested
*   Visualize the data and test results
*   Plot training curves, metric histograms, example images

The code is organized into sections. The important sections that it makes sense to edit (or call) are:

> **User Parameters** - Edit important user parameters and decide what the code will do
> **(!) Run Pipeline** - Run cell to tune, train, or test networks, or visualize dataset.

Colab GPUs
====
From best to worst:
V100 - 6.92/hr
L4 - 2.15/hr
T4 - 1.7/hr
v6e-1 TPU - 4.21/hr
v5e-1 TPU - 4.11/hr
v2-8 TPU - 1.82/hr
"""

#####################
### General Setup ###
#####################
## Basic Options ##
run_mode='none'  # Options: 'tune' , 'train' , 'test' , 'visualize' , 'none' ('none' builds dictionaries like you are visualizing but does not visualize)
network_type='ACT'    # 'ACT', 'ATTEN', 'CONCAT', 'FROZEN_COFLOW', 'FROZEN_COUNTERFLOW' (Unmaintained: 'GAN', 'CYCLEGAN', 'SIMULT')
train_SI=True         # If working with GAN or SUP networks, set to True build Sinogram-->Image networks, or False for Image --> Sinogram.
use_cache=False   # Cache dataset to Google Colab VM? Uses time to copy files. Might make dataset faster, might not.
cache_max_gb = 40 # Max size for cache. You'll error if you go over this.
cache_dir = '/content/cache'

plot_mode='inline'    # Options: 'always' (always show plots), 'inline' (only in Jupyter/Interactive Window), 'never' (silent)

## See note below for info about these options ##
gen_sino_size=288         # Options: 180, 288, 320. Resize input sinograms to this size. Sinograms are square, which was found to give the best results.
gen_image_size=180        # Image size (Options: 90). Images are square.
gen_sino_channels=3       # Number of sinogram channels for network currently being trained. (usually 1 or 3)
gen_image_channels=1      # Number of image channels for network currently being trained (generally 1)

SI_normalize=False    # For sino-->image mappings: normalize CNN outputs (images), iterative recons, and ground truths from dataset. You can then adjust the scale factor in the search dictionaries.
IS_normalize=False    # For image-->sinogram mappings: normalize CNN outputs (sinograms), projections, and ground truth sinograms from dataset. You can then adjust scale factor in search dicts.

## Scales ##
act_recon1_scale = 3.350  # If doing quantitative recons (no normalization), this is the scale factor to multiply optional recon1 by
act_recon2_scale = 1.998  # If doing quantitative recons (no normalization), this is the scale factor to multiply optional recon2 by
act_sino_scale   = 0.342  # If not normalizing sinograms, multiply sinograms by this factor
act_image_scale  = 1      # If not normalizing images, multiply by this factor.
                          # Set to 60 if you want to scale up activity maps to roughly equal counts/voxel (for our dataset).
atten_image_scale = 308.335  # Scale factor to multiply attenuation images by
atten_sino_scale = 39.187140258382726 # Scale factor to multiply attenuation sinograms by

## Resources ##
# Resources With Which to Initialize Ray Tune #
num_CPUs=12 # T4:8 L4/V100: 12
num_GPUs=1

# Set Resources Allocated per Trial #
CPUs_per_trial=2
GPUs_per_trial=1
device_opt='cuda' # Options: 'sense', cuda', 'cpu'. Set to 'sense' to set to 'cpu' if available, else 'cpu'.

# Colab / Local Setup Options ##
skip_local_package_installs=True # Local only: skip package checks and pip installs, just reload code. Set to True after first run for faster startup.
ray_tune_version=None # Optional: Pin Ray Tune to specific version (e.g., '2.9.0'). Set to None to use latest.
skip_colab_git_update=False # Colab only: Skip git pull when setting up repository. Useful if you have uncommitted changes or git operations fail.
setup_mode='walk' # Local only: 'walk' for module walking or 'install' for editable install from local repo.
base_repo_path=None # Optional: override repo root detection (useful for notebooks).

## Github Repository for Functions & Classes ##
github_username='petercl8'
repo_name='FlexCNN_for_Medical_Physics'

## Directories ##
project_colab_dirPath = '/content/drive/MyDrive/Colab/Working/'     # Directory, relative to which all other directories are specified (if working on Colab)
project_local_dirPath = r'C:\Users\Peter Lindstrom\My Drive (lindstrom.peter@gmail.com)\Colab\Working'  # Directory, relative to which all other directories are specified (if working Locally)

local_repo_dirPath =  r'C:\FlexCNN_cloned'

data_dirName = 'dataset-sets'      # Dataset directory, placed in project directory (above)
plot_dirName=  'plots'             # Plots Directory, placed in project directory (above)
checkpoint_dirName='checkpoints'   # If not using Ray Tune (not tuning), PyTorch saves and loads checkpoint file from here
                                   # All checkpoint files (for training, testing, visualizing) save the states for a particular network.
                                   # Therefore, the hyperparameters for the loaded CNN must match the data in the checkpoint file.
num_examples=-1                    # Number of examples from dataset to load. Set to -1 to use all examples (this is the default)


# NOTE: The concatenation network type introduces a fundamental problem: the sinogram input to the generator has channel numbers not corresponding to either the attenuation or the activity sinogram. My solution is to let the dataloader handle its own business (detect data structure sizes, since it has access to them) but let the user determine the number ofgenerator channels, since the generator does not see the data until after it has been instantiated. Also, the user only determines channels for currently trained network. Any frozen networks are always an attenuation network, so I hardcode channel information for dataloader and generator accordingly. Conclusion: sino_channels and image_channels represent the input and output channels for the generator for the currently trained network only.


############
## Tuning ##
############
# Note: When tuning, ALWAYS select "restart session and run all" from Runtime menu in Google Colab, or there may be bugs.
tune_csv_file='frame-ACT-288-padZeros-tunedSSIM' # .csv file to save tuning dataframe to
#tune_csv_file='temp'

tune_exp_name='search-ACT-288-padZeros-tunedSSIM'  # Experiment directory: Ray tune (and Tensorboard) write to this directory, relative to tune_storage_dirName.
#tune_exp_name='temp'

tune_scheduler = 'ASHA'      # Use FIFO for simple first in/first out to train to the end, or ASHA to early stop poorly performing trials.
tune_dataframe_fraction=0.33 # The fraction of the max tuning steps (tune_max_t) at which to save values to the tuning dataframe.
tune_restore=False           # Resume a terminated run (loads tune_exp_name from tune_storage_dirPath). If False, deletes any existing tune_exp_name folder and starts fresh.
tune_minutes = 8*60           # How long to run RayTune. 180 minutes is good for 180x180 input.
tune_metric = 'SSIM'   # Tune for which optimization metric? For val set: 'MSE', 'SSIM', 'CUSTOM' (user defined in the code). For QA set: 'CR_symmetric', 'hot_underestimation', 'cold_overestimation'
tune_even_reporting=True     # Set to True to ensure we report to Raytune at an even number of training examples, regardless of batch size.
tune_batches_per_report=10   # If tune_even_reporting = False, this is the number of batches per report (15 works pretty well).
tune_examples_per_report=4*256 # If tune_even_reporting = True, this is the number of training examples per Raytune report (4*512 = 1048 is a good number)
tune_grace_period=4          # Minimum number of reports before terminating a trial
tune_max_t = 24              # Maximum number of reports before terminating a trial
                             # 24 is a good number for ASHA. For FIFO, 12 is a good number.
tune_report_for='val'        # Set to 'val' to report IQA metrics using or cross-validation set. Set to 'qa' to use contrast recovery coefficients for QA phantoms.
tune_eval_batch_size=64   # If tuning on validation or QA set, what is the batch size to evaluate?
tune_augment=('SI', True)    # 'SI' (sinogram-->image or image--sinogram), "II" (image-->image) or None; True/False = augument by flipping along channels dimension?
tune_debug=False             # Run logger to debug tuning
tune_force_fixed_config=False# Force tuning with a fixed configuration dictionary. This is useful for debugging, to make sure that a network has a good architecture for learning.
tune_search_alg='optuna'     # 'optuna' or 'hyperopt'

## Training Files ##
## -------------- ##
#tune_act_sino_file ='train-highCountSino-382x513.npy'
tune_act_sino_file='train-highCountSino-320x257.npy'
#tune_act_sino_file='train-highCountImage.npy'
#tune_act_sino_file='train-obliqueImage.npy'
tune_act_image_file='train-actMap.npy'

tune_atten_sino_file=None
#tune_atten_sino_file='train-attenSino-382x513.npy'
tune_atten_image_file=None
#tune_atten_image_file='train-attenMap.npy'

tune_act_recon1_file=None
#act_recon1_file='train-highCountImage.npy' # Can set recon files to None if dataset does not have these.
tune_act_recon2_file=None
#tune_act_recon2_file='train-obliqueImage.npy'

## Cross Validation Set ##
## -------------------- ##
#tune_val_act_sino_file='val-highCountSino-382x513.npy'
tune_val_act_sino_file='val-highCountSino-320x257.npy'
#tune_val_act_sino_file='val-highCountImage.npy'
#tune_val_act_sino_file='val-obliqueImage.npy'
tune_val_act_image_file='val-actMap.npy'

tune_val_atten_image_file=None
tune_val_atten_sino_file=None


# QA Phantoms #
tune_qa_hot_weight=0.5 # A weighted contrast recovery coefficient is reported to ray tune as follows: ROI_NEMA_hot * tune_qa_hot_weight + ROI_NEMA_cold * (1-tune_qa_hot_weight)

tune_qa_act_sino_file=None #'QA-NEMA-highCountSino.npy'
tune_qa_act_image_file=None #'QA-NEMA-actMap.npy'
tune_qa_atten_image_file=None
tune_qa_atten_sino_file=None

tune_qa_hotMask_file=None #'QA-NEMA-hotMask_17mm.npy'
tune_qa_hotBackgroundMask_file=None #'QA-NEMA-backMask_17mm.npy'
tune_qa_coldMask_file=None #'QA-NEMA-coldMask_37mm.npy'
tune_qa_coldBackgroundMask_file=None #'QA-NEMA-backMask_37mm.npy'



## Unlikely to Change ##
tune_storage_dirName='searches'     # Create tuning folders (one for each experiment, each of which contains multiple trials) in this directory. Leave blank ('') to place search files in project directory
tune_dataframe_dirName= 'dataframes-tune'  # Directory for tuning dataframe (stores network information for each network trialed). Code will create it if it doesn't exist.

"""## Training"""

##############
## Training ##
##############

# Note: For dual network training, checkpoints are autmatically appended suffixes of -atten and -act.

#train_checkpoint_file='checkpoint-fullSet-highCountSino2actMap-tunedLDM-augSI-100epochs' # Checkpoint file to load or save to.
train_checkpoint_file='temp'

train_load_state=False     # Set to True to load pretrained weights. Use if training terminated early.
train_save_state=False     # Save network weights to train_checkpoint_file file as it trains
train_epochs = 200         # Number of training epochs.
train_display_step=10      # Number of steps/visualization. Good values: for supervised learning or GAN, set to: 50, For cycle-consistent, set to 20
train_sample_division=1    # To evenly sample the training set by a given factor, set this to an integer greater than 1 (ex: to sample every other example, set to 2)
train_show_times=True     # Show calculation times during training?


## Data Files & Augmentations ##
## -------------------------- ##
train_shuffle=True
train_augment=('SI', True)     # 'SI' (sinogram-->image or image--sinogram), "II" (image-->image) or None; True/False = augument by flipping along channels dimension?
#train_augment=('II', True)

#train_act_sino_file='train-highCountSino-382x513.npy'
train_act_sino_file='train-highCountSino-320x257.npy'
#train_act_sino_file='train-highCountImage.npy'

train_act_image_file='train-actMap.npy'
#train_act_image_file='train-anniMap.npy'

#rain_atten_image_file=None
train_atten_image_file='train-attenMap.npy'
train_atten_sino_file='train-attenSino-382x513.npy'

train_act_recon1_file=None
train_act_recon2_file=None
#train_act_recon1_file='train-highCountImage.npy'  # Can set recon files to None if dataset does not have these.
#train_act_recon2_file='train-obliqueImage.npy'
#train_act_recon1_file='train-actMap.npy'


###########
# Testing #
###########
test_dataframe_dirName= 'TestOnFull'  # Directory for test metric dataframes
test_csv_file = 'combined-tunedLowSSIM-trainedLowSSIM-onTestSet-wMLEM' # csv dataframe file to save testing results to
test_checkpoint_file='checkpoint-tunedLowSSIM-trainedLowSSIM-100epochs' # Checkpoint to load model for testing

test_display_step=15        # Make this a larger number to save bit of time (displays images/metrics less often)
test_batch_size=25          # This doesn't affect the final metrics, just the displayed metrics as testing procedes
test_chunk_size=875              # How many examples do you want to test at once? NOTE: This should be a multiple of test_batch_size AND also go into the test set size evenly.
testset_size=35000          # Size of the set to test. This must be <= the number of examples in your test set file.
test_begin_at=0             # Begin testing at this example number.
test_compute_MLEM=False          # Compute a simple MLEM reconstruction from the sinograms when running testing.
                            # This takes a lot longer. If set to false, only FBP is calculated.
test_merge_dataframes=True  # Merge the smaller/chunked dataframes at the end of the test run into one large dataframe?
test_show_times=False       # Show calculation times?
test_shuffle=False
test_sample_division=1

## Select Data Files ##
## ----------------- ##
test_act_sino_file= 'test-highCountSino-180x180.npy'
test_act_image_file= 'test-actMap.npy'

test_atten_image_file=None
test_atten_sino_file=None

test_act_recon1_file='test-highCountImage.npy'
test_act_recon2_file='test-obliqueImage.npy'

#test_act_sino_file=  'test_sino-35k.npy'
#test_act_image_file= 'test_image-35k.npy'
#test_act_sino_file= 'test_sino-highMSE-8750.npy'
#test_act_image_file= 'test_image-highMSE-8750.npy'
#test_act_sino_file= 'test_sino-lowMSE-8750.npy'
#test_act_image_file= 'test_image-lowMSE-8750.npy'


####################
## Visualize Data ##
####################
#visualize_act_sino_file= 'train-highCountSino-180x180.npy'
visualize_act_sino_file= 'train-highCountSino-382x513.npy'
visualize_act_image_file='train-actMap.npy'
visualize_act_recon1_file='train-highCountImage.npy'  # Can set recon files to None if dataset does not have these.
visualize_act_recon2_file='train-obliqueImage.npy'
#visualize_act_recon1_file=None
#visualize_act_recon2_file=None

visualize_atten_image_file=None
visualize_atten_sino_file=None

#visualize_checkpoint_file='checkpoint-90x1-tunedMSE-fc6-6epochs' # Checkpoint file to load/save
visualize_checkpoint_file='checkpoint-new_data-old_net-180x180-temp'
visualize_batch_size = 10   # Set value to exactly 120 to see a large grid of images OR =<10 for reconstructions
                            #  and ground truth with matched color scales
visualize_offset=0          # Image to begin at. Set to 0 to start at beginning.
visualize_shuffle=True      # Shuffle data set when visualizing?

"""# Install Required Packages

## Some Setup Funcs
"""

import os
import sys


# Minimal bootstrap helpers (no external imports, works before package is available)
def sense_colab():
    """Detect if running in Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def bootstrap_repo_root(base_repo_path=None):
    """Add repo root to sys.path so we can import the package."""
    if base_repo_path:
        repo_root = base_repo_path
    else:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
        repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    return repo_root


# Import setup helpers from the dedicated module (will be available after bootstrap)
def load_setup_environment():
    """Lazy-load setup helpers. Must be called AFTER bootstrap_repo_root()."""
    from FlexCNN_for_Medical_Physics.functions.setup_notebook.setup_environment import (
        install_packages,
        sense_device,
        setup_colab_environment,
        setup_local_environment,
    )
    return install_packages, sense_device, setup_colab_environment, setup_local_environment


# Bootstrap now so sys.path is ready
bootstrap_repo_root(base_repo_path=base_repo_path)

"""## Install Packages & Setup Repo"""

# Sense environment first
IN_COLAB = sense_colab()

if IN_COLAB:
    # Colab: setup repo FIRST (clone/install), then load setup helpers
    import subprocess
    import importlib
    
    base_dir = "/content"
    repo_path = os.path.join(base_dir, repo_name)
    repo_url = f"https://github.com/{github_username}/{repo_name}.git"
    
    # Clone repo if needed
    if not os.path.exists(repo_path):
        print(f"Cloning {repo_name}...")
        subprocess.run(["git", "clone", repo_url], cwd=base_dir, check=True)
    elif not skip_colab_git_update:
        print(f"Pulling latest changes...")
        try:
            subprocess.run(["git", "pull"], cwd=repo_path, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: git pull failed. Proceeding with existing repo...")
    
    # Install package in editable mode
    print("Installing package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
                   cwd=repo_path, check=True)
    
    # Explicitly add repo_path to sys.path for Colab (bootstrap_repo_root doesn't work in notebooks)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    
    # Force Python to forget old module cache
    if 'FlexCNN_for_Medical_Physics' in sys.modules:
        del sys.modules['FlexCNN_for_Medical_Physics']
    
    # Now it's safe to load and use setup helpers
    install_packages, sense_device, setup_colab_environment, setup_local_environment = load_setup_environment()
    
    # Install Python packages
    install_packages(IN_COLAB, ray_version=ray_tune_version)
    
    # Reload the package to pick up symbols
    package = importlib.import_module(repo_name)
    def reload_submodules(pkg):
        for _, modname, _ in __import__('pkgutil').walk_packages(pkg.__path__, pkg.__name__ + "."):
            try:
                importlib.reload(__import__(modname, fromlist=['']))
            except Exception:
                pass
    reload_submodules(package)
else:
    # Local: Load setup helpers (package is already on sys.path), then setup
    install_packages, sense_device, setup_colab_environment, setup_local_environment = load_setup_environment()
    
    # Install packages if needed
    if not skip_local_package_installs:
        install_packages(IN_COLAB, ray_version=ray_tune_version)
    else:
        print("Skipping package checks (skip_local_package_installs=True)")
    
    # Setup local environment
    setup_local_environment(repo_name=repo_name, mode=setup_mode, base_repo_path=base_repo_path)


# --- Test Resources ---
list_compute_resources()

# --- Set main project directory ---
project_dirPath = setup_project_dirs(IN_COLAB, project_local_dirPath, project_colab_dirPath, mount_colab_drive=True)

# --- Set Device ---
device = sense_device(device=device_opt)

# --- Configure Plotting ---
from FlexCNN_for_Medical_Physics.functions.helper.display_images import configure_plotting
configure_plotting(plot_mode=plot_mode)

# Build grouped parameter dictionaries #

common_settings = {
    'run_mode': run_mode,
    'device': device,
    'use_cache': use_cache,
    'cache_max_gb': cache_max_gb,
    'cache_dir': cache_dir,
    'num_examples': num_examples,
    'act_recon1_scale': act_recon1_scale,
    'act_recon2_scale': act_recon2_scale,
    'act_sino_scale': act_sino_scale,
    'act_image_scale': act_image_scale,
    'atten_image_scale': atten_image_scale,
    'atten_sino_scale': atten_sino_scale,
}

base_dirs = {
    'project_dirPath': project_dirPath,
    'plot_dirName': plot_dirName,
    'checkpoint_dirName': checkpoint_dirName,
    'tune_storage_dirName': tune_storage_dirName,
    'tune_dataframe_dirName': tune_dataframe_dirName,
    'test_dataframe_dirName': test_dataframe_dirName,
    'data_dirName': data_dirName
}

data_files = {
    'tune_act_sino_file': tune_act_sino_file,
    'tune_act_image_file': tune_act_image_file,
    'tune_act_recon1_file': tune_act_recon1_file,
    'tune_act_recon2_file': tune_act_recon2_file,
    'tune_atten_image_file': tune_atten_image_file,
    'tune_atten_sino_file': tune_atten_sino_file,
    'tune_val_act_sino_file': tune_val_act_sino_file,
    'tune_val_act_image_file': tune_val_act_image_file,
    'tune_val_atten_image_file': tune_val_atten_image_file,
    'tune_val_atten_sino_file': tune_val_atten_sino_file,
    'tune_qa_act_sino_file': tune_qa_act_sino_file,
    'tune_qa_act_image_file': tune_qa_act_image_file,
    'tune_qa_hotMask_file': tune_qa_hotMask_file,
    'tune_qa_hotBackgroundMask_file': tune_qa_hotBackgroundMask_file,
    'tune_qa_coldMask_file': tune_qa_coldMask_file,
    'tune_qa_coldBackgroundMask_file': tune_qa_coldBackgroundMask_file,
    'tune_qa_backMask_file': None, # Added this line to fix the KeyError
    'tune_qa_atten_image_file': tune_qa_atten_image_file,
    'tune_qa_atten_sino_file': tune_qa_atten_sino_file,
    'train_act_sino_file': train_act_sino_file,
    'train_act_image_file': train_act_image_file,
    'train_act_recon1_file': train_act_recon1_file,
    'train_act_recon2_file': train_act_recon2_file,
    'train_atten_image_file': train_atten_image_file,
    'train_atten_sino_file': train_atten_sino_file,
    'test_act_sino_file': test_act_sino_file,
    'test_act_image_file': test_act_image_file,
    'test_act_recon1_file': test_act_recon1_file,
    'test_act_recon2_file': test_act_recon2_file,
    'test_atten_image_file': test_atten_image_file,
    'test_atten_sino_file': test_atten_sino_file,
    'visualize_act_sino_file': visualize_act_sino_file,
    'visualize_act_image_file': visualize_act_image_file,
    'visualize_act_recon1_file': visualize_act_recon1_file,
    'visualize_act_recon2_file': visualize_act_recon2_file,
    'visualize_atten_image_file': visualize_atten_image_file,
    'visualize_atten_sino_file': visualize_atten_sino_file,
    }

mode_files = {
    'tune_csv_file': tune_csv_file,
    'train_checkpoint_file': train_checkpoint_file,
    'test_checkpoint_file': test_checkpoint_file,
    'test_csv_file': test_csv_file,
    'visualize_checkpoint_file': visualize_checkpoint_file
}

network_opts = {
    'network_type': network_type,
    'train_SI': train_SI,
    'gen_image_size': gen_image_size,
    'gen_sino_size': gen_sino_size,
    'gen_image_channels': gen_image_channels,
    'gen_sino_channels': gen_sino_channels,
    'SI_normalize': SI_normalize,
    'IS_normalize': IS_normalize,
}

tune_opts = {
    'tune_exp_name': tune_exp_name,
    'tune_scheduler': tune_scheduler,
    'tune_dataframe_fraction': tune_dataframe_fraction,
    'tune_restore': tune_restore,
    'tune_max_t': tune_max_t,
    'tune_minutes': tune_minutes,
    'tune_metric': tune_metric,
    'tune_even_reporting': tune_even_reporting,
    'tune_batches_per_report': tune_batches_per_report,
    'tune_examples_per_report': tune_examples_per_report,
    'tune_augment': tune_augment,
    'tune_grace_period': tune_grace_period,
    'tune_debug': tune_debug,
    'tune_force_fixed_config': tune_force_fixed_config,
    'tune_report_for': tune_report_for,
    'tune_qa_hot_weight': tune_qa_hot_weight,
    'tune_eval_batch_size': tune_eval_batch_size,
    'num_CPUs': num_CPUs,
    'num_GPUs': num_GPUs,
    'cpus_per_trial': CPUs_per_trial,  # per trial
    'gpus_per_trial': GPUs_per_trial,  # per trial
    'tune_search_alg': tune_search_alg,
}

train_opts = {
    'train_load_state': train_load_state,
    'train_save_state': train_save_state,
    'training_epochs': train_epochs,
    'train_augment': train_augment,
    'train_shuffle': train_shuffle,
    'train_display_step': train_display_step,
    'train_sample_division': train_sample_division,
    'train_show_times': train_show_times,
}

test_opts = {
    'test_display_step': test_display_step,
    'test_batch_size': test_batch_size,
    'test_chunk_size': test_chunk_size,
    'testset_size': testset_size,
    'test_begin_at': test_begin_at,
    'test_compute_MLEM': test_compute_MLEM,
    'test_merge_dataframes': test_merge_dataframes,
    'test_show_times': test_show_times,
    'test_shuffle': test_sample_division,
    'test_sample_division': test_sample_division
}

viz_opts = {
    'visualize_batch_size': visualize_batch_size,
    'visualize_offset': visualize_offset,
    'visualize_shuffle': visualize_shuffle,
}

# Build paths and settings using new functions
paths = setup_paths(
    run_mode=run_mode,
    base_dirs=base_dirs,
    data_files=data_files,
    mode_files=mode_files,
    test_ops=test_opts,
    viz_ops=viz_opts
)

settings = setup_settings(
    run_mode=run_mode,
    common_settings=common_settings,
    tune_opts=tune_opts,
    train_opts=train_opts,
    test_opts=test_opts,
    viz_opts=viz_opts,
)

# --- Build Config Dictionary ---
config = construct_config(
    run_mode=run_mode,
    network_opts=network_opts,
    tune_opts=tune_opts,
    test_opts=test_opts,
    viz_opts=viz_opts,
    config_ACT_SI=config_ACT_SI,
    config_ACT_IS=config_ACT_IS,
    config_ATTEN_SI=config_ATTEN_SI,
    config_ATTEN_IS=config_ATTEN_IS,
    config_CONCAT=config_CONCAT,
    config_FROZEN_COFLOW=config_FROZEN_COFLOW,
    config_FROZEN_COUNTERFLOW=config_FROZEN_COUNTERFLOW,
    config_GAN_SI=None,
    config_GAN_IS=None,
    config_RAY_SI=config_RAY_SI,
    config_RAY_SI_learnScale=config_RAY_SI_learnScale,
    config_RAY_SI_fixedScale=config_RAY_SI_fixedScale,
    config_RAY_IS=config_RAY_IS,
    config_RAY_IS_learnScale=config_RAY_IS_learnScale,
    config_RAY_IS_fixedScale=config_RAY_IS_fixedScale,
    config_RAY_SUP=config_RAY_SUP,
    config_RAY_SUP_FROZEN=config_RAY_SUP_FROZEN,
    config_RAY_GAN=None,
    config_RAY_GAN_CYCLE=None,
)

# --- Run Pipeline ---
run_pipeline(
    config=config,
    paths=paths,
    settings=settings,
    tune_opts=tune_opts,
    base_dirs=base_dirs,
    test_opts=test_opts,
    IN_COLAB=IN_COLAB
)