import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from .dataset_augment_data_recons import AugmentSinoImageDataRecons, AugmentImageImageDataRecons
from .dataset_resizing import resize_image_data, crop_pad_sino

resize_warned = False  # Module-level flag to ensure warning is printed only once



def NpArrayDataLoader(image_array, sino_array, config, settings, augment=False, sino_resize_type='crop_pad', sino_pad_type='sinogram', image_pad_type='none', index=0, device='cuda', recon1_array=None, recon2_array=None):
    global resize_warned
    '''
    Function to load a sinogram, activity map, and optionally reconstructions. Returns 4 pytorch tensors:
    sino_scaled, act_map_scaled, recon1, recon2 (both reconstructions may be None).

    image_array:         activity map numpy array (ground truth)
    sino_array:          sinogram numpy array
    config:              configuration dictionary with hyperparameters. Must contain: network_type, train_SI, image_size, 
                         sino_size, image_channels, sino_channels, SI_normalize, SI_fixedScale, and (for non-SUP/GAN networks) 
                         IS_normalize, SI_fixedScale.
    augment:             perform data augmentation?
    sino_resize_type:    'crop_pad' to crop/pad to target size, 'bilinear' to use bilinear interpolation
    sino_pad_type:       'sinogram' (default) to pad sinograms with sinogram-like padding, 'zeros' to pad with zeros
    image_pad_type:      'none' (default) to resize w/ bilinear interpolation to correct size, 'zeros' to pad with zeros without resizing     
    index:               index of the sinogram/activity map pair to grab
    device:              device to place tensors on ('cuda' or 'cpu')
    recon1_array:        (optional) reconstruction 1 numpy array
    recon2_array:        (optional) reconstruction 2 numpy array
    recon1_scale:        (optional) scaling factor for recon1 to match ground truth quantitatively
    recon2_scale:        (optional) scaling factor for recon2 to match ground truth quantitatively
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
    recon1_scale = settings.get('recon1_scale', 1.0)
    recon2_scale = settings.get('recon2_scale', 1.0)
    sino_scale = settings.get('sino_scale', 1.0)

    ## Select Data, Convert to Tensors ##
    act_map_multChannel = torch.from_numpy(np.ascontiguousarray(image_array[index,:])).float()
    sinogram_multChannel = torch.from_numpy(np.ascontiguousarray(sino_array[index,:])).float()
    recon1_multChannel = torch.from_numpy(np.ascontiguousarray(recon1_array[index,:])).float() if recon1_array is not None else None
    recon2_multChannel = torch.from_numpy(np.ascontiguousarray(recon2_array[index,:])).float() if recon2_array is not None else None

    ## Resize Warning ##
    orig_channel, orig_image_h, orig_image_w = act_map_multChannel.shape
    orig_sino_channel, orig_sino_h, orig_sino_w = sinogram_multChannel.shape

    resize_sino = (orig_sino_h, orig_sino_w) != (sino_size, sino_size)
    resize_image = (orig_image_h, orig_image_w) != (image_size, image_size) or (recon1_multChannel is not None and (recon1_multChannel.shape[1], recon1_multChannel.shape[2]) != (image_size, image_size)) or (recon2_multChannel is not None and (recon2_multChannel.shape[1], recon2_multChannel.shape[2]) != (image_size, image_size))

    if not resize_warned:
        if resize_sino or resize_image:
            print(f"Warning: Dataset resized. Original image size: ({orig_channel}, {orig_image_h}, {orig_image_w}), target: ({image_channels}, {image_size}, {image_size}). "
                  f"Original sinogram size: ({orig_sino_channel}, {orig_sino_h}, {orig_sino_w}), target: ({sino_channels}, {sino_size}, {sino_size}).")
            resize_warned = True


    #### AUGMENT AND RESIZE DATA ####

    ## Augment Sinograms ##
    if augment[0]=='SI':
        # We augment first so that the sinogram columns, which contain the full span of angles, are not truncated before sinogram-like augmentation.
        act_map_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel = AugmentSinoImageDataRecons(
            act_map_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel, flip_channels=augment[1]
        )
        # Resize sinogram (like a Sinogram)
        if resize_sino:
            if sino_resize_type=='bilinear':
                sinogram_multChannel_resize = transforms.Resize(size=(sino_size, sino_size), antialias=True)(sinogram_multChannel)
            else:
                sinogram_multChannel_resize = crop_pad_sino(sinogram_multChannel, vert_size=sino_size, target_width=sino_size, pool_size=2, pad_type=sino_pad_type)
        else:
            sinogram_multChannel_resize = sinogram_multChannel

    if augment[0]=='II':
        # If doing image-like augmentations, first resize sinogram (like an Image). This way, rotations are not truncated.
        if resize_sino:
            if sino_resize_type=='bilinear':
                sinogram_multChannel_resize = transforms.Resize(size=(sino_size, sino_size), antialias=True)(sinogram_multChannel)
            else: # For image inputs, pad_type and pool_size are hardcoded to 1 and 'zeros' (the only sensible options for image-like sinograms)
                sinogram_multChannel_resize = crop_pad_sino(sinogram_multChannel, vert_size=sino_size, target_width=sino_size, pool_size=2, pad_type='zeros')
        else:
            sinogram_multChannel_resize = sinogram_multChannel

        # Augment data (with image-like augmentations)
        act_map_multChannel, sinogram_multChannel_resize, recon1_multChannel, recon2_multChannel = AugmentImageImageDataRecons(
            act_map_multChannel, sinogram_multChannel_resize, recon1_multChannel, recon2_multChannel, flip_channels=augment[1]
        )

    # Resize image data (only if needed)
    act_map_multChannel_resize, recon1_multChannel_resize, recon2_multChannel_resize = resize_image_data(
        act_map_multChannel, recon1_multChannel, recon2_multChannel, image_size, resize_image=resize_image, image_pad_type=image_pad_type
    )

    #### (Optional) Normalize Resized Outputs ####
    if SI_normalize:
        a = torch.reshape(act_map_multChannel_resize, (image_channels,-1))
        a = nn.functional.normalize(a, p=1, dim = 1)
        act_map_multChannel_resize = torch.reshape(a, (image_channels, image_size, image_size))
        if recon1_multChannel_resize is not None:
            b = torch.reshape(recon1_multChannel_resize, (image_channels,-1))
            b = nn.functional.normalize(b, p=1, dim = 1)
            recon1_multChannel_resize = torch.reshape(b, (image_channels, image_size, image_size))
        if recon2_multChannel_resize is not None:
            c = torch.reshape(recon2_multChannel_resize, (image_channels,-1))
            c = nn.functional.normalize(c, p=1, dim = 1)
            recon2_multChannel_resize = torch.reshape(c, (image_channels, image_size, image_size))
    if IS_normalize:
        a = torch.reshape(sinogram_multChannel_resize, (sino_channels,-1))
        a = nn.functional.normalize(a, p=1, dim = 1)
        sinogram_multChannel_resize = torch.reshape(a, (sino_channels, sino_size, sino_size))


    #### Scale and Move to Device ####

    # If SI_normalize==True: multiply activity and reconstructions by SI_fixedScale (recon scales already set to 1.0)
    # If SI_normalize==False: leave activity unchanged; multiply reconstructions by their respective recon scales
    if SI_normalize:
        act_map_scaled = (SI_fixedScale * act_map_multChannel_resize).to(device)
        recon1 = (SI_fixedScale * recon1_multChannel_resize).to(device) if recon1_multChannel_resize is not None else None
        recon2 = (SI_fixedScale * recon2_multChannel_resize).to(device) if recon2_multChannel_resize is not None else None
    else:
        act_map_scaled = act_map_multChannel_resize.to(device)
        recon1 = (recon1_multChannel_resize * recon1_scale).to(device) if recon1_multChannel_resize is not None else None
        recon2 = (recon2_multChannel_resize * recon2_scale).to(device) if recon2_multChannel_resize is not None else None

    ## Apply Fixed Scales per desired behavior and move to device ##
    # Sinogram: multiply by IS_fixedScale only if IS_normalize==True; otherwise multiply by sino_scale if not normalized
    if IS_normalize:
        sino_scaled = (IS_fixedScale * sinogram_multChannel_resize).to(device)
    else:
        sino_scaled = (sinogram_multChannel_resize * sino_scale).to(device)

    return sino_scaled, act_map_scaled, recon1, recon2




class NpArrayDataSet(Dataset):
    '''
    Class for loading data from .np files, given file directory strings and set of optional transformations.
    In the dataset used in our first two conference papers, the data repeat every 17500 steps but with different augmentations.
    For the dataset with FORE rebinning, the dataset contains no augmented examples; all augmentation is performed on the fly.
    '''
    def __init__(self, image_path, sino_path, config, settings, augment=False, offset=0, num_examples=-1, sample_division=1, device='cuda', recon1_path=None, recon2_path=None):
        '''
        image_path:         path to images (ground truth activity maps) in data set
        sino_path:          path to sinograms in data set
        config:             configuration dictionary with hyperparameters. Must contain: image_size, sino_size, 
                            image_channels, sino_channels, network_type, train_SI, SI_normalize, SI_fixedScale, 
                            and (for non-SUP/GAN networks) IS_normalize, SI_fixedScale.
        settings:           dictionary containing recon1_scale, recon2_scale, sino_scale, etc.
        augment:            Set True to perform on-the-fly augmentation of data set. Set False to not perform augmentation.
        offset:             To begin dataset at beginning of the datafile, set offset=0. To begin on the second image, offset = 1, etc.
        num_examples:       Max number of examples to load into dataset. Set to -1 to load the maximum number from the numpy array.
        sample_division:    set to 1 to use every example, 2 to use every other example, etc. (Ex: if sample_division=2, the dataset will be half the size.)
        recon1_path:        (optional) path to pre-computed reconstruction 1. If None, reconstructions will be computed on-the-fly.
        recon2_path:        (optional) path to pre-computed reconstruction 2. If None, reconstructions will be computed on-the-fly.
        '''

        ## Load Data to Arrays ##
        image_array = np.load(image_path, mmap_mode='r')
        sino_array = np.load(sino_path, mmap_mode='r')
        recon1_array = np.load(recon1_path, mmap_mode='r') if recon1_path is not None else None
        recon2_array = np.load(recon2_path, mmap_mode='r') if recon2_path is not None else None

        ## Set Instance Variables ##
        if num_examples==-1:
            self.image_array = image_array[offset:,:]
            self.sino_array = sino_array[offset:,:]
            self.recon1_array = recon1_array[offset:,:] if recon1_array is not None else None
            self.recon2_array = recon2_array[offset:,:] if recon2_array is not None else None
        else:
            self.image_array = image_array[offset : offset + num_examples, :]
            self.sino_array = sino_array[offset : offset + num_examples, :]
            self.recon1_array = recon1_array[offset : offset + num_examples, :] if recon1_array is not None else None
            self.recon2_array = recon2_array[offset : offset + num_examples, :] if recon2_array is not None else None

        self.config = config
        self.settings = settings
        self.augment = augment
        self.sample_division = sample_division
        self.device = device
        self.recon1_path = recon1_path
        self.recon2_path = recon2_path

    def __len__(self):
        length = int(len(self.image_array)/self.sample_division)
        return length

    def __getitem__(self, idx):
        idx = idx*self.sample_division
        device_arg = self.device
        if device_arg == 'cuda' and not torch.cuda.is_available():
            device_arg = 'cpu'
        sino_scaled, act_map_scaled, recon1, recon2 = NpArrayDataLoader(
            self.image_array, self.sino_array, self.config, self.settings,
            augment=self.augment, index=idx, device=device_arg,
            recon1_array=self.recon1_array, recon2_array=self.recon2_array)

        # Only return reconstructions if they exist (to avoid collate errors with None values)
        if recon1 is not None and recon2 is not None:
            return sino_scaled, act_map_scaled, recon1, recon2
        elif recon1 is not None:
            return sino_scaled, act_map_scaled, recon1
        elif recon2 is not None:
            return sino_scaled, act_map_scaled, recon2
        else:
            return sino_scaled, act_map_scaled
