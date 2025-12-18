import numpy as np
import torch
import torchvision.transforms as transforms
from .augment_data import IntersectSquareBorder


def AugmentSinoImageDataRecons(image_multChannel, sinogram_multChannel, recon1_multChannel=None, recon2_multChannel=None, flip_channels=False):
    ## Data Augmentation Functions ##
    def RandRotateSinoImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel):
        '''
        Function for randomly rotating an image and its sinogram. If the image intersects the edge of the FOV, no rotation is applied.

        image_multChannel:    image to rotate. Shape: (C, H, W)
        sinogram_multChannel: sinogram to rotate. Shape: (C, H, W)
        recon1_multChannel:   optional recon 1 to rotate. Shape: (C, H, W)
        recon2_multChannel:   optional recon 2 to rotate. Shape: (C, H, W)
        '''    
        if IntersectSquareBorder(image_multChannel) == False:
            bins = sinogram_multChannel.shape[2]
            bins_shifted = np.random.randint(0, bins)
            angle = int(bins_shifted * 180/bins)

            image_multChannel = transforms.functional.rotate(image_multChannel, angle, fill=0) # Rotate image. Fill in unspecified pixels with zeros.
            if recon1_multChannel is not None:
                recon1_multChannel = transforms.functional.rotate(recon1_multChannel, angle, fill=0) # Rotate recon1.
            if recon2_multChannel is not None:
                recon2_multChannel = transforms.functional.rotate(recon2_multChannel, angle, fill=0) # Rotate recon2.
            sinogram_multChannel = torch.roll(sinogram_multChannel, bins_shifted, dims=(2,)) # Cycle (or 'Roll') sinogram by that angle along dimension 2.
            sinogram_multChannel[:,:, 0:bins_shifted] = torch.flip(sinogram_multChannel[:,:,0:bins_shifted], dims=(1,)) # flip the cycled portion of the sinogram vertically

        return image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel

    def VerticalFlipSinoImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel):
        image_multChannel = torch.flip(image_multChannel,dims=(1,)) # Flip image vertically
        sinogram_multChannel = torch.flip(sinogram_multChannel,dims=(1,2)) # Flip sinogram horizontally and vertically
        if recon1_multChannel is not None:
            recon1_multChannel = torch.flip(recon1_multChannel, dims=(1,)) # Flip recon1 vertically
        if recon2_multChannel is not None:
            recon2_multChannel = torch.flip(recon2_multChannel, dims=(1,)) # Flip recon2 vertically
        return image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel

    def HorizontalFlipSinoImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel):
        image_multChannel = torch.flip(image_multChannel, dims=(2,)) # Flip image horizontally
        sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(2,)) # Flip sinogram horizontally
        if recon1_multChannel is not None:
            recon1_multChannel = torch.flip(recon1_multChannel, dims=(2,)) # Flip recon1 horizontally
        if recon2_multChannel is not None:
            recon2_multChannel = torch.flip(recon2_multChannel, dims=(2,)) # Flip recon2 horizontally
        return image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel

    def ChannelFlipSinoImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel):
        sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(0,)) # Flip sinogram channels about center channel
        # Leave recon1/recon2 unchanged to match original semantics for channel flip
        return image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel

    image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel = RandRotateSinoImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel)           # Always rotates image by a random angle
    if np.random.choice([True, False]): # Half of the time, flips the image vertically
        image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel = VerticalFlipSinoImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel)
    if np.random.choice([True, False]): # Half of the time, flips the image horizontally
        image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel = HorizontalFlipSinoImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel)
    if flip_channels==True and np.random.choice([True, False]): # Half of the time, flips the sinogram channels about the center channel
        image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel = ChannelFlipSinoImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel)

    return image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel


def AugmentImageImageDataRecons(image_multChannel, sinogram_multChannel, recon1_multChannel=None, recon2_multChannel=None, flip_channels=False):
    ## If one would like to train a network that goes from image to image (e.g. an image denoising network),
    ## this function can be used to augment the paired image data. To keep nomenclature consistent, the sinogram input is still called 'sinogram_multChannel' here.
    ## Data Augmentation Functions ##
    def RandRotateImageImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel):
        '''
        Function for randomly rotating two paired images. If the image intersects the edge of the FOV, no rotation is applied.
        '''    
        if IntersectSquareBorder(image_multChannel) == False:
            angle = np.random.randint(0,360)
            image_multChannel = transforms.functional.rotate(image_multChannel, angle, fill=0) # Rotate image. Fill in unspecified pixels with zeros.
            sinogram_multChannel = transforms.functional.rotate(sinogram_multChannel, angle, fill=0) # Rotate sinogram. Fill in unspecified pixels with zeros.
            if recon1_multChannel is not None:
                recon1_multChannel = transforms.functional.rotate(recon1_multChannel, angle, fill=0) # Rotate recon1.
            if recon2_multChannel is not None:
                recon2_multChannel = transforms.functional.rotate(recon2_multChannel, angle, fill=0) # Rotate recon2.

        return image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel

    def VerticalFlipImageImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel):
        image_multChannel = torch.flip(image_multChannel,dims=(1,)) # Flip image vertically
        sinogram_multChannel = torch.flip(sinogram_multChannel,dims=(1,)) # Flip sinogram vertically
        if recon1_multChannel is not None:
            recon1_multChannel = torch.flip(recon1_multChannel, dims=(1,)) # Flip recon1 vertically
        if recon2_multChannel is not None:
            recon2_multChannel = torch.flip(recon2_multChannel, dims=(1,)) # Flip recon2 vertically
        return image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel

    def HorizontalFlipImageImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel):
        image_multChannel = torch.flip(image_multChannel, dims=(2,)) # Flip image horizontally
        sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(2,)) # Flip sinogram horizontally
        if recon1_multChannel is not None:
            recon1_multChannel = torch.flip(recon1_multChannel, dims=(2,)) # Flip recon1 horizontally
        if recon2_multChannel is not None:
            recon2_multChannel = torch.flip(recon2_multChannel, dims=(2,)) # Flip recon2 horizontally
        return image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel

    def ChannelFlipImageImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel):
        image_multChannel = torch.flip(image_multChannel, dims=(0,)) # Flip image channels about center channel
        sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(0,)) # Flip sinogram channels about center channel
        if recon1_multChannel is not None:
            recon1_multChannel = torch.flip(recon1_multChannel, dims=(0,)) # Flip recon1 channels about center channel
        if recon2_multChannel is not None:
            recon2_multChannel = torch.flip(recon2_multChannel, dims=(0,)) # Flip recon2 channels about center channel
        return image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel

    image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel = RandRotateImageImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel)           # Always rotates image by a random angle
    if np.random.choice([True, False]): # Half of the time, flips the image vertically
        image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel = VerticalFlipImageImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel)
    if np.random.choice([True, False]): # Half of the time, flips the image horizontally
        image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel = HorizontalFlipImageImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel)
    if flip_channels==True and np.random.choice([True, False]): # Half of the time, flips the sinogram channels about the center channel
        image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel = ChannelFlipImageImage(image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel)

    return image_multChannel, sinogram_multChannel, recon1_multChannel, recon2_multChannel
