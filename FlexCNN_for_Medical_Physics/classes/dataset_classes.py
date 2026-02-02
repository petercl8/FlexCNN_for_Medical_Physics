# ========================================================================================
# DATASET CLASSES - Data Loading and Augmentation for Training/Testing
# ========================================================================================
# This module provides two core components:
# 1. NpArrayDataLoader: Function to load and preprocess individual data samples with optional augmentation
# 2. NpArrayDataSet: PyTorch Dataset class for loading data from .np files with transformations

import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from .dataset_augment_data_recons import AugmentSinoImageDataRecons, AugmentImageImageDataRecons
from .dataset_resizing import resize_image_data, resize_sino_data

resize_warned = False  # Module-level flag to ensure warning is printed only once


# ========================================================================================
# FUNCTION: NpArrayDataLoader
# ========================================================================================
# Load and preprocess a single data sample. Returns (act_data, atten_data, recon_data) where:
#   act_data = (act_sino_scaled, act_image_scaled)
#   atten_data = (atten_sino_scaled, atten_image_scaled)
#   recon_data = (act_recon1, act_recon2)
# Any entry may be None depending on network type.
#
# NETWORK TYPE REQUIREMENTS:
#   ACT, FROZEN_COFLOW, FROZEN_COUNTERFLOW: Require activity data; atten_* can be None
#   ATTEN: Requires attenuation data; act_sino/act_image can be None
#   CONCAT: Requires both activity and attenuation data
#   (Path validation in trainable.py enforces requirements before calling this loader)
#
# PARAMETERS:
#   act_sino_array:      activity sinogram array (None for ATTEN networks)
#   act_image_array:     activity image array (None for ATTEN networks)
#   atten_image_array:   attenuation image array (None for ACT networks)
#   atten_sino_array:    attenuation sinogram array (None for ACT networks)
#   act_recon1_array:    (optional) reconstruction 1 array
#   act_recon2_array:    (optional) reconstruction 2 array
#   config:              configuration dictionary with network_type, train_SI, gen_image_size, gen_sino_size,
#                        gen_image_channels, gen_sino_channels, SI_normalize, SI_fixedScale, IS_normalize, IS_fixedScale
#   augment:             augmentation type: 'SI', 'II', None, or False
#   sino_resize_type:    'crop_pad' (default) or 'bilinear'
#   sino_pad_type:       'sinogram' (default) or 'zeros'
#   image_pad_type:      'zeros' (default--padds original-sized image with surrounding zeros) or 'none' (bilinear resize)
#   index:               data sample index to extract
#   device:              'cuda' or 'cpu'

def NpArrayDataLoader(act_sino_array, act_image_array, atten_image_array, atten_sino_array, act_recon1_array, act_recon2_array,
                      config, settings, augment=False,
                      sino_resize_type='bilinear', sino_pad_type='zeros', image_pad_type='zeros', index=0, device='cuda',
                      ):

    global resize_warned
    # ========================================================================================
    # SECTION 1: Extract Parameters from Configuration
    # ========================================================================================
    network_type = config['network_type']
    train_SI = config['train_SI']
    gen_image_size = config['gen_image_size']
    gen_sino_size = config['gen_sino_size']
    gen_image_channels = config['gen_image_channels']
    gen_sino_channels = config['gen_sino_channels']

    # ========================================================================================
    # SECTION 2: Set Normalization Variables
    # ========================================================================================
    # Normalization direction (SI vs IS) depends on network and training mode
    if network_type in ('GAN', 'ATTEN', 'ACT', 'CONCAT', 'FROZEN_COFLOW', 'FROZEN_COUNTERFLOW'): # supervisory network
        if train_SI==True:
            SI_normalize=config['SI_normalize']
            SI_fixedScale=config['SI_fixedScale']
            IS_normalize=False
            IS_fixedScale=1
        else:
            SI_normalize=False
            SI_fixedScale=1
            IS_normalize=config['IS_normalize']
            IS_fixedScale=config['IS_fixedScale']
    else: # cycle-consistent network
        SI_normalize=config['SI_normalize']
        SI_fixedScale=config['SI_fixedScale']
        IS_normalize=config['IS_normalize']
        IS_fixedScale=config['IS_fixedScale']

    # Extract scale factors from settings
    act_recon1_scale = settings['act_recon1_scale']
    act_recon2_scale = settings['act_recon2_scale']
    act_sino_scale = settings['act_sino_scale']
    act_image_scale = settings['act_image_scale']
    atten_image_scale = settings['atten_image_scale']
    atten_sino_scale = settings['atten_sino_scale']

    # ========================================================================================
    # SECTION 3: Select Data and Convert to Tensors
    # ========================================================================================
    # Activity data (required for ACT/CONCAT/FROZEN; None for ATTEN)
    act_image_multChannel = torch.from_numpy(np.ascontiguousarray(act_image_array[index,:])).float() if act_image_array is not None else None
    act_sino_multChannel = torch.from_numpy(np.ascontiguousarray(act_sino_array[index,:])).float() if act_sino_array is not None else None
    act_recon1_multChannel = torch.from_numpy(np.ascontiguousarray(act_recon1_array[index,:])).float() if act_recon1_array is not None else None
    act_recon2_multChannel = torch.from_numpy(np.ascontiguousarray(act_recon2_array[index,:])).float() if act_recon2_array is not None else None

    # Attenuation data (required for ATTEN/CONCAT/FROZEN; None for ACT)
    atten_image_multChannel = torch.from_numpy(np.ascontiguousarray(atten_image_array[index,:])).float() if atten_image_array is not None else None
    atten_sino_multChannel = torch.from_numpy(np.ascontiguousarray(atten_sino_array[index,:])).float() if atten_sino_array is not None else None

    # ========================================================================================
    # SECTION 4: Resize Warning
    # ========================================================================================
    # Check if sinograms need resizing
    resize_sino = ((act_sino_multChannel is not None and act_sino_multChannel.shape[1:] != (gen_sino_size, gen_sino_size)) or
                   (atten_sino_multChannel is not None and atten_sino_multChannel.shape[1:] != (gen_sino_size, gen_sino_size)))
    
    # Check if images need resizing
    resize_image = ((act_image_multChannel is not None and act_image_multChannel.shape[1:] != (gen_image_size, gen_image_size)) or
                    (atten_image_multChannel is not None and atten_image_multChannel.shape[1:] != (gen_image_size, gen_image_size)) or
                    (act_recon1_multChannel is not None and act_recon1_multChannel.shape[1:] != (gen_image_size, gen_image_size)) or
                    (act_recon2_multChannel is not None and act_recon2_multChannel.shape[1:] != (gen_image_size, gen_image_size)))
    
    # Detect channel counts for logging/normalization
    act_image_channels_det = act_image_multChannel.shape[0] if act_image_multChannel is not None else None
    act_recon1_channels_det = act_recon1_multChannel.shape[0] if act_recon1_multChannel is not None else None
    act_recon2_channels_det = act_recon2_multChannel.shape[0] if act_recon2_multChannel is not None else None
    atten_image_channels_det = atten_image_multChannel.shape[0] if atten_image_multChannel is not None else None
    act_sino_channels_det = act_sino_multChannel.shape[0] if act_sino_multChannel is not None else None
    atten_sino_channels_det = atten_sino_multChannel.shape[0] if atten_sino_multChannel is not None else None

    # Print resize warning once with detected channels
    if not resize_warned and (resize_sino or resize_image):
        print(
            "Warning: Dataset will be resized to image size: "
            f"({act_image_channels_det or gen_image_channels}, {gen_image_size}, {gen_image_size}), "
            "sinogram size: "
            f"({act_sino_channels_det or gen_sino_channels}, {gen_sino_size}, {gen_sino_size}). "
            "Note: channels are auto-detected from data when present."
        )
        resize_warned = True


    # ========================================================================================
    # SECTION 5: Augment and Resize Data
    # ========================================================================================

    # Initialize resized sinograms to originals to avoid unbound variables when no augmentation path runs
    act_sino_multChannel_resize = act_sino_multChannel
    atten_sino_multChannel_resize = atten_sino_multChannel
    
    if augment[0]=='SI':
        # We augment first so that the sinogram columns, which contain the full span of angles, are not truncated before sinogram-like augmentation.
        act_sino_multChannel, act_image_multChannel, atten_sino_multChannel, atten_image_multChannel, act_recon1_multChannel, act_recon2_multChannel = AugmentSinoImageDataRecons(
            act_sino_multChannel, act_image_multChannel, atten_sino_multChannel, atten_image_multChannel, act_recon1_multChannel, act_recon2_multChannel, flip_channels=augment[1]
        )
        # Resize sinogram (like a Sinogram)
        act_sino_multChannel_resize, atten_sino_multChannel_resize = resize_sino_data(
            act_sino_multChannel, atten_sino_multChannel, gen_sino_size,
            resize_sino=resize_sino, sino_resize_type=sino_resize_type, sino_pad_type=sino_pad_type, pool_size=2
        )

    if augment[0]=='II':
        # If doing image-like augmentations, first resize sinogram (like an Image). This way, rotations are not truncated.
        act_sino_multChannel_resize, atten_sino_multChannel_resize = resize_sino_data(
            act_sino_multChannel, atten_sino_multChannel, gen_sino_size,
            resize_sino=resize_sino, sino_resize_type=sino_resize_type, sino_pad_type='zeros', pool_size=2
        )

        # Augment data (with image-like augmentations)
        act_sino_multChannel_resize, act_image_multChannel, atten_sino_multChannel_resize, atten_image_multChannel, act_recon1_multChannel, act_recon2_multChannel = AugmentImageImageDataRecons(
            act_sino_multChannel_resize, act_image_multChannel, atten_sino_multChannel_resize, atten_image_multChannel, act_recon1_multChannel, act_recon2_multChannel, flip_channels=augment[1]
        )

    if augment[0] is None:
        act_sino_multChannel_resize, atten_sino_multChannel_resize = resize_sino_data(
            act_sino_multChannel, atten_sino_multChannel, gen_sino_size,
            resize_sino=resize_sino, sino_resize_type=sino_resize_type, sino_pad_type=sino_pad_type, pool_size=2
        )

    # Resize image data (only if needed)
    act_image_multChannel_resize, atten_image_multChannel_resize, act_recon1_multChannel_resize, act_recon2_multChannel_resize = resize_image_data(
        act_image_multChannel, atten_image_multChannel, act_recon1_multChannel, act_recon2_multChannel, gen_image_size, resize_image=resize_image, image_pad_type=image_pad_type
    )

    # ========================================================================================
    # SECTION 6: Normalize Resized Outputs (Optional)
    # ========================================================================================
    # Activity image/sino normalized when SI_normalize=True (typically ACT/FROZEN networks)
    if SI_normalize:
        if act_image_multChannel_resize is not None:
            a = torch.reshape(act_image_multChannel_resize, (act_image_channels_det,-1))
            a = nn.functional.normalize(a, p=1, dim = 1)
            act_image_multChannel_resize = torch.reshape(a, (act_image_channels_det, gen_image_size, gen_image_size))
        if act_recon1_multChannel_resize is not None:
            b = torch.reshape(act_recon1_multChannel_resize, (act_recon1_channels_det,-1))
            b = nn.functional.normalize(b, p=1, dim = 1)
            act_recon1_multChannel_resize = torch.reshape(b, (act_recon1_channels_det, gen_image_size, gen_image_size))
        if act_recon2_multChannel_resize is not None:
            c = torch.reshape(act_recon2_multChannel_resize, (act_recon2_channels_det,-1))
            c = nn.functional.normalize(c, p=1, dim = 1)
            act_recon2_multChannel_resize = torch.reshape(c, (act_recon2_channels_det, gen_image_size, gen_image_size))
        if atten_image_multChannel_resize is not None:
            d = torch.reshape(atten_image_multChannel_resize, (atten_image_channels_det,-1))
            d = nn.functional.normalize(d, p=1, dim = 1)
            atten_image_multChannel_resize = torch.reshape(d, (atten_image_channels_det, gen_image_size, gen_image_size))
    if IS_normalize:
        if act_sino_multChannel_resize is not None:
            a = torch.reshape(act_sino_multChannel_resize, (act_sino_channels_det,-1))
            a = nn.functional.normalize(a, p=1, dim = 1)
            act_sino_multChannel_resize = torch.reshape(a, (act_sino_channels_det, gen_sino_size, gen_sino_size))
        if atten_sino_multChannel_resize is not None:
            b = torch.reshape(atten_sino_multChannel_resize, (atten_sino_channels_det,-1))
            b = nn.functional.normalize(b, p=1, dim = 1)
            atten_sino_multChannel_resize = torch.reshape(b, (atten_sino_channels_det, gen_sino_size, gen_sino_size))


    # ========================================================================================
    # SECTION 7: Scale and Move to Device
    # ========================================================================================

    # If SI_normalize==True: multiply activity and reconstructions by SI_fixedScale (recon scales already set to 1.0)
    # If SI_normalize==False: leave activity unchanged; multiply reconstructions by their respective recon scales
    if SI_normalize:
        act_image_scaled = (SI_fixedScale * act_image_multChannel_resize).to(device) if act_image_multChannel_resize is not None else None
        act_recon1_scaled = (SI_fixedScale * act_recon1_multChannel_resize).to(device) if act_recon1_multChannel_resize is not None else None
        act_recon2_scaled = (SI_fixedScale * act_recon2_multChannel_resize).to(device) if act_recon2_multChannel_resize is not None else None
        atten_image_scaled = (SI_fixedScale * atten_image_multChannel_resize).to(device) if atten_image_multChannel_resize is not None else None
    else:
        act_image_scaled = (act_image_scale * act_image_multChannel_resize).to(device) if act_image_multChannel_resize is not None else None
        act_recon1_scaled = (act_recon1_scale * act_recon1_multChannel_resize).to(device) if act_recon1_multChannel_resize is not None else None
        act_recon2_scaled = (act_recon2_scale * act_recon2_multChannel_resize).to(device) if act_recon2_multChannel_resize is not None else None
        atten_image_scaled = (atten_image_scale * atten_image_multChannel_resize).to(device) if atten_image_multChannel_resize is not None else None

    # Apply fixed scales per desired behavior and move to device
    # Sinogram: multiply by IS_fixedScale only if IS_normalize==True; otherwise multiply by sino_scale if not normalized
    if IS_normalize:
        act_sino_scaled = (IS_fixedScale * act_sino_multChannel_resize).to(device) if act_sino_multChannel_resize is not None else None
        atten_sino_scaled= (IS_fixedScale * atten_sino_multChannel_resize).to(device) if atten_sino_multChannel_resize is not None else None
    else:
        act_sino_scaled = (act_sino_scale * act_sino_multChannel_resize).to(device) if act_sino_multChannel_resize is not None else None
        atten_sino_scaled = (atten_sino_scale * atten_sino_multChannel_resize).to(device) if atten_sino_multChannel_resize is not None else None

    # Return nested tuple structure: (act_data, atten_data, recon_data)
    act_data = (act_sino_scaled, act_image_scaled)  # activity sinogram/image
    atten_data = (atten_sino_scaled, atten_image_scaled)
    recon_data = (act_recon1_scaled, act_recon2_scaled)  # activity reconstructions

    return act_data, atten_data, recon_data


# ========================================================================================
# CLASS: NpArrayDataSet
# ========================================================================================
# PyTorch Dataset class for loading data from .np files with optional transformations.
# In the dataset used in our first two conference papers, the data repeat every 17500 steps but with different augmentations.
# For the dataset with FORE rebinning, the dataset contains no augmented examples; all augmentation is performed on the fly.

class NpArrayDataSet(Dataset):

    # ========================================================================================
    # METHOD: __init__
    # ========================================================================================
    # Initialize dataset. Pass None for paths not required by your network type.
    #
    # PARAMETERS:
    #   act_sino_path:      path to activity sinograms (None for ATTEN networks)
    #   act_image_path:     path to activity images (None for ATTEN networks)
    #   atten_image_path:   path to attenuation images (None for ACT networks)
    #   atten_sino_path:    path to attenuation sinograms (None for ACT networks)
    #   act_recon1_path:    (optional) reconstruction 1 path
    #   act_recon2_path:    (optional) reconstruction 2 path
    #   config:             configuration dict with network_type, gen_image_size, gen_sino_size, normalization settings, etc.
    #   settings:           dict with scale factors (act_recon1_scale, act_sino_scale, etc.)
    #   augment:            augmentation type ('SI', 'II', False)
    #   offset:             start index in dataset
    #   num_examples:       max examples to load (-1 for all)
    #   sample_division:    step size for sampling (1=all, 2=every other, etc.)
    #   device:             'cuda' or 'cpu'
    #
    # Examples:
    #   ACT network:     act_sino_path, act_image_path, atten_image_path=None, atten_sino_path=None
    #   ATTEN network:   act_sino_path=None, act_image_path=None, atten_image_path, atten_sino_path
    #   CONCAT network:  act_sino_path, act_image_path, atten_image_path, atten_sino_path

    def __init__(self, act_sino_path, act_image_path, atten_image_path, atten_sino_path, act_recon1_path, act_recon2_path, config, settings, augment=False, offset=0, num_examples=-1, sample_division=1, device='cuda'):

        # ========================================================================================
        # SUBSECTION: Load Data to Arrays
        # ========================================================================================
        image_array = np.load(act_image_path, mmap_mode='r') if act_image_path is not None else None
        sino_array = np.load(act_sino_path, mmap_mode='r') if act_sino_path is not None else None
        recon1_array = np.load(act_recon1_path, mmap_mode='r') if act_recon1_path is not None else None
        recon2_array = np.load(act_recon2_path, mmap_mode='r') if act_recon2_path is not None else None
        atten_image_array = np.load(atten_image_path, mmap_mode='r') if atten_image_path is not None else None
        atten_sino_array = np.load(atten_sino_path, mmap_mode='r') if atten_sino_path is not None else None

        # ========================================================================================
        # SUBSECTION: Set Instance Variables
        # ========================================================================================
        if num_examples==-1:
            self.image_array = image_array[offset:,:] if image_array is not None else None
            self.sino_array = sino_array[offset:,:] if sino_array is not None else None
            self.recon1_array = recon1_array[offset:,:] if recon1_array is not None else None
            self.recon2_array = recon2_array[offset:,:] if recon2_array is not None else None
            self.atten_image_array = atten_image_array[offset:,:] if atten_image_array is not None else None
            self.atten_sino_array = atten_sino_array[offset:,:] if atten_sino_array is not None else None
        else:
            self.image_array = image_array[offset : offset + num_examples, :] if image_array is not None else None
            self.sino_array = sino_array[offset : offset + num_examples, :] if sino_array is not None else None
            self.recon1_array = recon1_array[offset : offset + num_examples, :] if recon1_array is not None else None
            self.recon2_array = recon2_array[offset : offset + num_examples, :] if recon2_array is not None else None
            self.atten_image_array = atten_image_array[offset : offset + num_examples, :] if atten_image_array is not None else None
            self.atten_sino_array = atten_sino_array[offset : offset + num_examples, :] if atten_sino_array is not None else None

        self.config = config
        self.settings = settings
        self.augment = augment
        self.sample_division = sample_division
        self.device = device
        self.recon1_path = act_recon1_path
        self.recon2_path = act_recon2_path
        self.atten_image_path = atten_image_path
        self.atten_sino_path = atten_sino_path

    def __len__(self):
        # Use first non-None array to determine length
        for arr in [self.image_array, self.sino_array, self.atten_image_array, self.atten_sino_array, self.recon1_array, self.recon2_array]:
            if arr is not None:
                length = int(len(arr)/self.sample_division)
                return length
        return 0  # All arrays are None

    def __getitem__(self, idx):
        idx = idx*self.sample_division

        device_arg = self.device
        if device_arg == 'cuda' and not torch.cuda.is_available():
            device_arg = 'cpu'

        act_data, atten_data, recon_data = NpArrayDataLoader(
            self.sino_array, self.image_array, self.atten_image_array, self.atten_sino_array,
            self.recon1_array, self.recon2_array,
            self.config, self.settings,
            augment=self.augment, index=idx, device=device_arg)

        return act_data, atten_data, recon_data
