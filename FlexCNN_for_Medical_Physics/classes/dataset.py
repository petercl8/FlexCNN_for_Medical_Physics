import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from .augment_data import AugmentSinoImageData, AugmentImageImageData

resize_warned = False  # Module-level flag to ensure warning is printed only once

class NpArrayDataSet(Dataset):
    '''
    Class for loading data from .np files, given file directory strings and set of optional transformations.
    In the dataset used in our first two conference papers, the data repeat every 17500 steps but with different augmentations.
    For the dataset with FORE rebinning, the dataset contains no augmented examples; all augmentation is performed on the fly.
    '''
    def __init__(self, image_path, sino_path, config, augment=False, offset=0, num_examples=-1, sample_division=1, device='cuda', FORE_recon_path=None, oblique_recon_path=None):
        '''
        image_path:         path to images (ground truth activity maps) in data set
        sino_path:          path to sinograms in data set
        config:             configuration dictionary with hyperparameters. Must contain: image_size, sino_size, 
                            image_channels, sino_channels, network_type, train_SI, SI_normalize, SI_fixedScale, 
                            and (for non-SUP/GAN networks) IS_normalize, SI_fixedScale.
        augment:            Set True to perform on-the-fly augmentation of data set. Set False to not perform augmentation.
        offset:             To begin dataset at beginning of the datafile, set offset=0. To begin on the second image, offset = 1, etc.
        num_examples:       Max number of examples to load into dataset. Set to -1 to load the maximum number from the numpy array.
        sample_division:    set to 1 to use every example, 2 to use every other example, etc. (Ex: if sample_division=2, the dataset will be half the size.)
        FORE_recon_path:    (optional) path to pre-computed FORE reconstructions. If None, reconstructions will be computed on-the-fly.
        oblique_recon_path: (optional) path to pre-computed oblique reconstructions. If None, reconstructions will be computed on-the-fly.
        '''

        ## Load Data to Arrays ##
        image_array = np.load(image_path, mmap_mode='r')       # We use memmaps to significantly speed up the loading.
        sino_array = np.load(sino_path, mmap_mode='r')
        FORE_recon_array = np.load(FORE_recon_path, mmap_mode='r') if FORE_recon_path is not None else None
        oblique_recon_array = np.load(oblique_recon_path, mmap_mode='r') if oblique_recon_path is not None else None

        ## Set Instance Variables ##
        if num_examples==-1:
            self.image_array = image_array[offset:,:]
            self.sino_array = sino_array[offset:,:]
            self.FORE_recon_array = FORE_recon_array[offset:,:] if FORE_recon_array is not None else None
            self.oblique_recon_array = oblique_recon_array[offset:,:] if oblique_recon_array is not None else None
        else:
            self.image_array = image_array[offset : offset + num_examples, :]
            self.sino_array = sino_array[offset : offset + num_examples, :]
            self.FORE_recon_array = FORE_recon_array[offset : offset + num_examples, :] if FORE_recon_array is not None else None
            self.oblique_recon_array = oblique_recon_array[offset : offset + num_examples, :] if oblique_recon_array is not None else None

        self.config = config
        self.augment = augment
        self.sample_division = sample_division
        self.device = device

    def __len__(self):
        length = int(len(self.image_array)/self.sample_division)
        return length

    def __getitem__(self, idx):

        idx = idx*self.sample_division

        device_arg = self.device
        if device_arg == 'cuda' and not torch.cuda.is_available():
            device_arg = 'cpu'
        sino_scaled, act_map_scaled, FORE_recon, oblique_recon = NpArrayDataLoader(
            self.image_array, self.sino_array, self.config,
            augment=self.augment, index=idx, device=device_arg,
            FORE_recon_array=self.FORE_recon_array, oblique_recon_array=self.oblique_recon_array)

        return sino_scaled, act_map_scaled, FORE_recon, oblique_recon


def NpArrayDataLoader(image_array, sino_array, config, augment=False, index=0, device='cuda', FORE_recon_array=None, oblique_recon_array=None):
    
    global resize_warned

    '''
    Function to load a sinogram, activity map, and optionally reconstructions. Returns 4 pytorch tensors:
    sino_scaled, act_map_scaled, FORE_recon, oblique_recon (both reconstructions may be None).

    image_array:         activity map numpy array (ground truth)
    sino_array:          sinogram numpy array
    config:              configuration dictionary with hyperparameters. Must contain: network_type, train_SI, image_size, 
                         sino_size, image_channels, sino_channels, SI_normalize, SI_fixedScale, and (for non-SUP/GAN networks) 
                         IS_normalize, SI_fixedScale.
    augment:             perform data augmentation?
    index:               index of the sinogram/activity map pair to grab
    device:              device to place tensors on ('cuda' or 'cpu')
    FORE_recon_array:    (optional) FORE reconstruction numpy array
    oblique_recon_array: (optional) oblique reconstruction numpy array
    '''
    ## Extract parameters from config ##
    network_type = config['network_type']
    train_SI = config['train_SI']
    image_size = config['image_size']
    sino_size = config['sino_size']
    image_channels = config['image_channels']
    sino_channels = config['sino_channels']
    
    ## Set Normalization Variables ##
    if (network_type=='GAN') or (network_type=='SUP'):
        if train_SI==True:
            SI_normalize=config['SI_normalize']
            SI_fixedScale=config['SI_fixedScale']
            IS_normalize=False     # If the Sinogram-->Image network (SI) is being trained, IS normalization is not in the config dict.
            IS_fixedScale=1             # If the Sinogram-->Image network (SI) is being trained, IS scaling is not in the config dict.
        else:
            SI_normalize=False
            SI_fixedScale=1
            IS_normalize=config['IS_normalize']
            IS_fixedScale=config['IS_fixedScale']

    else: # If a cycle-consistent network, normalize & scale everything
        SI_normalize=config['SI_normalize']
        SI_fixedScale=config['SI_fixedScale']
        IS_normalize=config['IS_normalize']
        IS_fixedScale=config['IS_fixedScale']


    ## Select Data, Convert to Pytorch Tensors ##
    act_map_multChannel = torch.from_numpy(image_array[index,:]).float() # act_map_multChannel.shape = (C, X, Y)
    sinogram_multChannel = torch.from_numpy(sino_array[index,:]).float() # sinogram_multChannel.shape = (C, X, Y)
    FORE_recon_multChannel = torch.from_numpy(FORE_recon_array[index,:]).float() if FORE_recon_array is not None else None
    oblique_recon_multChannel = torch.from_numpy(oblique_recon_array[index,:]).float() if oblique_recon_array is not None else None

    ## Run Data Augmentation on Selected Data. ##
    if augment[0]=='SI':
        act_map_multChannel, sinogram_multChannel = AugmentSinoImageData(act_map_multChannel, sinogram_multChannel, flip_channels=augment[1])
    if augment[0]=='II':
        act_map_multChannel, sinogram_multChannel = AugmentImageImageData(act_map_multChannel, sinogram_multChannel, flip_channels=augment[1])

    ## Create A Set of Resized Outputs ##
    orig_image_h, orig_image_w = act_map_multChannel.shape[1:]
    orig_sino_h, orig_sino_w = sinogram_multChannel.shape[1:]
    orig_FORE_h, orig_FORE_w = FORE_recon_multChannel.shape[1:] if FORE_recon_multChannel is not None else (None, None)
    orig_oblique_h, orig_oblique_w = oblique_recon_multChannel.shape[1:] if oblique_recon_multChannel is not None else (None, None)

    sinogram_multChannel_resize = transforms.Resize(size = (sino_size, sino_size), antialias=True)(sinogram_multChannel)
    act_map_multChannel_resize    = transforms.Resize(size = (image_size, image_size), antialias=True)(act_map_multChannel)
    FORE_recon_multChannel_resize = transforms.Resize(size = (image_size, image_size), antialias=True)(FORE_recon_multChannel) if FORE_recon_multChannel is not None else None
    oblique_recon_multChannel_resize = transforms.Resize(size = (image_size, image_size), antialias=True)(oblique_recon_multChannel) if oblique_recon_multChannel is not None else None

    # Warn if resizing changes dimensions
    if not resize_warned:
        if (orig_image_h, orig_image_w) != (image_size, image_size) or (orig_sino_h, orig_sino_w) != (sino_size, sino_size):
            print(f"Warning: Dataset resized. Original image size: ({orig_image_h}, {orig_image_w}), target: ({image_size}, {image_size}). "
                  f"Original sinogram size: ({orig_sino_h}, {orig_sino_w}), target: ({sino_size}, {sino_size}).")
            resize_warned = True
        if FORE_recon_multChannel is not None and (orig_FORE_h, orig_FORE_w) != (image_size, image_size):
            print(f"Warning: FORE reconstruction resized. Original size: ({orig_FORE_h}, {orig_FORE_w}), target: ({image_size}, {image_size}).")
        if oblique_recon_multChannel is not None and (orig_oblique_h, orig_oblique_w) != (image_size, image_size):
            print(f"Warning: Oblique reconstruction resized. Original size: ({orig_oblique_h}, {orig_oblique_w}), target: ({image_size}, {image_size}).")

    ## (Optional) Normalize Resized Outputs Along Channel Dimension ##
    if SI_normalize:
        a = torch.reshape(act_map_multChannel_resize, (image_channels,-1))
        a = nn.functional.normalize(a, p=1, dim = 1)
        act_map_multChannel_resize = torch.reshape(a, (image_channels, image_size, image_size))
        if FORE_recon_multChannel_resize is not None:
            b = torch.reshape(FORE_recon_multChannel_resize, (image_channels,-1))
            b = nn.functional.normalize(b, p=1, dim = 1)
            FORE_recon_multChannel_resize = torch.reshape(b, (image_channels, image_size, image_size))
        if oblique_recon_multChannel_resize is not None:
            c = torch.reshape(oblique_recon_multChannel_resize, (image_channels,-1))
            c = nn.functional.normalize(c, p=1, dim = 1)
            oblique_recon_multChannel_resize = torch.reshape(c, (image_channels, image_size, image_size))
    if IS_normalize:
        a = torch.reshape(sinogram_multChannel_resize, (sino_channels,-1))                     # Flattens each sinogram. Each channel is normalized.
        a = nn.functional.normalize(a, p=1, dim = 1)                      # Normalizes along dimension 1 (values for each of the 3 channels)
        sinogram_multChannel_resize = torch.reshape(a, (sino_channels, sino_size, sino_size))  # Reshapes sinograms back into squares.

    ## Adjust Output Channels of Resized Outputs ##
    if image_channels==1:
        act_map_out = act_map_multChannel_resize                 # For image_channels = 1, the image is just left alone
    else:
        act_map_out = act_map_multChannel_resize.repeat(image_channels,1,1)   # This could be altered to account for RGB images, etc.

    if sino_channels==1:
        sino_out = sinogram_multChannel_resize[0:1,:]        # Selects 1st sinogram channel only. Using 0:1 preserves the channels dimension.
    else:
        sino_out = sinogram_multChannel_resize               # Keeps full sinogram with all channels

    # Handle reconstruction channel adjustment
    if FORE_recon_multChannel_resize is not None:
        if image_channels==1:
            FORE_recon_out = FORE_recon_multChannel_resize
        else:
            FORE_recon_out = FORE_recon_multChannel_resize.repeat(image_channels,1,1)
    else:
        FORE_recon_out = None

    if oblique_recon_multChannel_resize is not None:
        if image_channels==1:
            oblique_recon_out = oblique_recon_multChannel_resize
        else:
            oblique_recon_out = oblique_recon_multChannel_resize.repeat(image_channels,1,1)
    else:
        oblique_recon_out = None

    # Apply scaling if normalized
    sino_scaled = IS_fixedScale * sino_out if IS_normalize else sino_out
    act_map_scaled = SI_fixedScale * act_map_out if SI_normalize else act_map_out
    FORE_recon_scaled = SI_fixedScale * FORE_recon_out if (SI_normalize and FORE_recon_out is not None) else FORE_recon_out
    oblique_recon_scaled = SI_fixedScale * oblique_recon_out if (SI_normalize and oblique_recon_out is not None) else oblique_recon_out

    return sino_scaled.to(device), act_map_scaled.to(device), FORE_recon_scaled.to(device) if FORE_recon_scaled is not None else None, oblique_recon_scaled.to(device) if oblique_recon_scaled is not None else None