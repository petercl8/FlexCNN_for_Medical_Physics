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
    def __init__(self, image_path, sino_path, config, augment=False, offset=0, num_examples=-1, sample_division=1, device='cuda'):
        '''
        image_path:         path to images in data set
        sino_path:          path to sinograms in data set
        config:             configuration dictionary with hyperparameters. Must contain: image_size, sino_size, 
                            image_channels, sino_channels, network_type, train_SI, SI_normalize, SI_fixedScale, 
                            and (for non-SUP/GAN networks) IS_normalize, SI_fixedScale.
        augment:            Set True to perform on-the-fly augmentation of data set. Set False to not perform augmentation.
        offset:             To begin dataset at beginning of the datafile, set offset=0. To begin on the second image, offset = 1, etc.
        num_examples:       Max number of examples to load into dataset. Set to -1 to load the maximum number from the numpy array.
        sample_division:    set to 1 to use every example, 2 to use every other example, etc. (Ex: if sample_division=2, the dataset will be half the size.)
        '''

        ## Load Data to Arrays ##
        image_array = np.load(image_path, mmap_mode='r')       # We use memmaps to significantly speed up the loading.
        sino_array = np.load(sino_path, mmap_mode='r')

        ## Set Instance Variables ##
        if num_examples==-1:
            self.image_array = image_array[offset:,:]
            self.sino_array = sino_array[offset:,:]
        else:
            self.image_array = image_array[offset : offset + num_examples, :]
            self.sino_array = sino_array[offset : offset + num_examples, :]

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
        sino_ground, sino_ground_scaled, image_ground, image_ground_scaled = NpArrayDataLoader(
            self.image_array, self.sino_array, self.config,
            augment=self.augment, index=idx, device=device_arg)

        return sino_ground, sino_ground_scaled, image_ground, image_ground_scaled
        # Returns both original, as well as altered, sinograms and images


def NpArrayDataLoader(image_array, sino_array, config, augment=False, index=0, device='cuda'):
    
    global resize_warned

    '''
    Function to load an image and a sinogram. Returns 4 pytorch tensors: the original dataset sinogram and image,
    and scaled and (optionally) normalized sinograms and images.

    image_array:    image numpy array
    sino_array:     sinogram numpy array
    config:         configuration dictionary with hyperparameters. Must contain: network_type, train_SI, image_size, 
                    sino_size, image_channels, sino_channels, SI_normalize, SI_fixedScale, and (for non-SUP/GAN networks) 
                    IS_normalize, SI_fixedScale.
    augment:        perform data augmentation?
    index:          index of the image/sinogram pair to grab
    device:         device to place tensors on ('cuda' or 'cpu')
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
    image_multChannel = torch.from_numpy(image_array[index,:]).float() # image_multChannel.shape = (C, X, Y)
    sinogram_multChannel = torch.from_numpy(sino_array[index,:]).float() # sinogram_multChannel.shape = (C, X, Y)

    ## Run Data Augmentation on Selected Data. ##
    if augment[0]=='SI':
        image_multChannel, sinogram_multChannel = AugmentSinoImageData(image_multChannel, sinogram_multChannel, flip_channels=augment[1])
    if augment[0]=='II':
        image_multChannel, sinogram_multChannel = AugmentImageImageData(image_multChannel, sinogram_multChannel, flip_channels=augment[1])

    ## Create A Set of Resized Outputs ##
    orig_image_h, orig_image_w = image_multChannel.shape[1:]
    orig_sino_h, orig_sino_w = sinogram_multChannel.shape[1:]

    sinogram_multChannel_resize = transforms.Resize(size = (sino_size, sino_size), antialias=True)(sinogram_multChannel)
    image_multChannel_resize    = transforms.Resize(size = (image_size, image_size), antialias=True)(image_multChannel)

    # Warn if resizing changes dimensions
    if not resize_warned:
        if (orig_image_h, orig_image_w) != (image_size, image_size) or (orig_sino_h, orig_sino_w) != (sino_size, sino_size):
            print(f"Warning: Dataset resized. Original image size: ({orig_image_h}, {orig_image_w}), target: ({image_size}, {image_size}). "
                  f"Original sinogram size: ({orig_sino_h}, {orig_sino_w}), target: ({sino_size}, {sino_size}).")
            resize_warned = True

    ## (Optional) Normalize Resized Outputs Along Channel Dimension ##
    if SI_normalize:
        a = torch.reshape(image_multChannel_resize, (image_channels,-1))
        a = nn.functional.normalize(a, p=1, dim = 1)
        image_multChannel_resize = torch.reshape(a, (image_channels, image_size, image_size))
    if IS_normalize:
        a = torch.reshape(sinogram_multChannel_resize, (sino_channels,-1))                     # Flattens each sinogram. Each channel is normalized.
        a = nn.functional.normalize(a, p=1, dim = 1)                      # Normalizes along dimension 1 (values for each of the 3 channels)
        sinogram_multChannel_resize = torch.reshape(a, (sino_channels, sino_size, sino_size))  # Reshapes sinograms back into squares.

    ## Adjust Output Channels of Resized Outputs ##
    if image_channels==1:
        image_out = image_multChannel_resize                 # For image_channels = 1, the image is just left alone
    else:
        image_out = image_multChannel_resize.repeat(image_channels,1,1)   # This chould be altered to account for RGB images, etc.

    if sino_channels==1:
        sino_out = sinogram_multChannel_resize[0:1,:]        # Selects 1st sinogram channel only. Using 0:1 preserves the channels dimension.
    else:
        sino_out = sinogram_multChannel_resize               # Keeps full sinogram with all channels

    # Returns both original and altered sinograms and images, assigned to CPU or GPU
    sino_scaled = IS_fixedScale * sino_out if IS_normalize else sino_out
    image_scaled = SI_fixedScale * image_out if SI_normalize else image_out

    return sinogram_multChannel.to(device), sino_scaled.to(device), image_multChannel.to(device), image_scaled.to(device)