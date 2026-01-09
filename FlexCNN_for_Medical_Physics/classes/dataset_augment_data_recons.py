import numpy as np
import torch
import torchvision.transforms as transforms


def AugmentSinoImageDataRecons(sinogram_multChannel, image_multChannel, atten_sino_multChannel=None, atten_image_multChannel=None, recon1_multChannel=None, recon2_multChannel=None, flip_channels=False):
    ## Data Augmentation Functions ##
    def RandRotateSinoImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel):
        '''
        Function for randomly rotating an image and its sinogram. If the image intersects the edge of the FOV, no rotation is applied.

        image_multChannel:    image to rotate. Shape: (C, H, W)
        sinogram_multChannel: sinogram to rotate. Shape: (C, H, W)
        atten_sino_multChannel: optional atten_sino to rotate. Shape: (C, H, W)
        atten_image_multChannel: optional atten_image to rotate. Shape: (C, H, W)
        recon1_multChannel:   optional recon 1 to rotate. Shape: (C, H, W)
        recon2_multChannel:   optional recon 2 to rotate. Shape: (C, H, W)
        '''    
        # Find first available image-like data to check FOV border
        test_image = None
        for img in [image_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel]:
            if img is not None:
                test_image = img
                break
        
        # Find first available sinogram-like data to get bins
        test_sino = None
        for sino in [sinogram_multChannel, atten_sino_multChannel]:
            if sino is not None:
                test_sino = sino
                break
        
        # Only rotate if we have data and it doesn't intersect border
        if test_image is not None and test_sino is not None and IntersectSquareBorder(test_image) == False:
            bins = test_sino.shape[2]
            bins_shifted = np.random.randint(0, bins)
            angle = int(bins_shifted * 180/bins)

            # Rotate all image-like data
            if image_multChannel is not None:
                image_multChannel = transforms.functional.rotate(image_multChannel, angle, fill=0)
            if atten_image_multChannel is not None:
                atten_image_multChannel = transforms.functional.rotate(atten_image_multChannel, angle, fill=0)
            if recon1_multChannel is not None:
                recon1_multChannel = transforms.functional.rotate(recon1_multChannel, angle, fill=0)
            if recon2_multChannel is not None:
                recon2_multChannel = transforms.functional.rotate(recon2_multChannel, angle, fill=0)
            
            # Roll and flip all sinogram-like data
            if sinogram_multChannel is not None:
                sinogram_multChannel = torch.roll(sinogram_multChannel, bins_shifted, dims=(2,))
                sinogram_multChannel[:,:, 0:bins_shifted] = torch.flip(sinogram_multChannel[:,:,0:bins_shifted], dims=(1,))
            if atten_sino_multChannel is not None:
                atten_sino_multChannel = torch.roll(atten_sino_multChannel, bins_shifted, dims=(2,))
                atten_sino_multChannel[:,:, 0:bins_shifted] = torch.flip(atten_sino_multChannel[:,:,0:bins_shifted], dims=(1,))

        return sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel

    def VerticalFlipSinoImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel):
        if sinogram_multChannel is not None:
            sinogram_multChannel = torch.flip(sinogram_multChannel,dims=(1,2)) # Flip sinogram horizontally and vertically
        if image_multChannel is not None:
            image_multChannel = torch.flip(image_multChannel,dims=(1,)) # Flip image vertically
        if atten_sino_multChannel is not None:
            atten_sino_multChannel = torch.flip(atten_sino_multChannel, dims=(1,2)) # Flip atten_sino horizontally and vertically
        if atten_image_multChannel is not None:
            atten_image_multChannel = torch.flip(atten_image_multChannel, dims=(1,)) # Flip atten_image vertically
        if recon1_multChannel is not None:
            recon1_multChannel = torch.flip(recon1_multChannel, dims=(1,)) # Flip recon1 vertically
        if recon2_multChannel is not None:
            recon2_multChannel = torch.flip(recon2_multChannel, dims=(1,)) # Flip recon2 vertically
        return sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel

    def HorizontalFlipSinoImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel):
        if sinogram_multChannel is not None:
            sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(2,)) # Flip sinogram horizontally
        if image_multChannel is not None:
            image_multChannel = torch.flip(image_multChannel, dims=(2,)) # Flip image horizontally
        if atten_sino_multChannel is not None:
            atten_sino_multChannel = torch.flip(atten_sino_multChannel, dims=(2,)) # Flip atten_sino horizontally
        if atten_image_multChannel is not None:
            atten_image_multChannel = torch.flip(atten_image_multChannel, dims=(2,)) # Flip atten_image horizontally
        if recon1_multChannel is not None:
            recon1_multChannel = torch.flip(recon1_multChannel, dims=(2,)) # Flip recon1 horizontally
        if recon2_multChannel is not None:
            recon2_multChannel = torch.flip(recon2_multChannel, dims=(2,)) # Flip recon2 horizontally
        return sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel

    def ChannelFlipSinoImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel):
        if sinogram_multChannel is not None:
            sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(0,)) # Flip sinogram channels about center channel
        if image_multChannel is not None:
            image_multChannel = torch.flip(image_multChannel, dims=(0,)) # Flip image channels about center channel
        if atten_sino_multChannel is not None:
            atten_sino_multChannel = torch.flip(atten_sino_multChannel, dims=(0,)) # Flip atten_sino channels about center channel
        if atten_image_multChannel is not None:
            atten_image_multChannel = torch.flip(atten_image_multChannel, dims=(0,)) # Flip atten_image channels about center channel
        if recon1_multChannel is not None:
            recon1_multChannel = torch.flip(recon1_multChannel, dims=(0,)) # Flip recon1 channels about center channel
        if recon2_multChannel is not None:
            recon2_multChannel = torch.flip(recon2_multChannel, dims=(0,)) # Flip recon2 channels about center channel
        return sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel

    sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel = RandRotateSinoImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel)           # Always rotates image by a random angle
    if np.random.choice([True, False]): # Half of the time, flips the image vertically
        sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel = VerticalFlipSinoImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel)
    if np.random.choice([True, False]): # Half of the time, flips the image horizontally
        sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel = HorizontalFlipSinoImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel)
    if flip_channels==True and np.random.choice([True, False]): # Half of the time, flips the sinogram channels about the center channel
        sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel = ChannelFlipSinoImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel)

    return sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel


def AugmentImageImageDataRecons(sinogram_multChannel, image_multChannel, atten_sino_multChannel=None, atten_image_multChannel=None, recon1_multChannel=None, recon2_multChannel=None, flip_channels=False):
    ## If one would like to train a network that goes from image to image (e.g. an image denoising network),
    ## this function can be used to augment the paired image data. To keep nomenclature consistent, the sinogram input is still called 'sinogram_multChannel' here.
    ## Data Augmentation Functions ##
    def RandRotateImageImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel):
        '''
        Function for randomly rotating two paired images. If the image intersects the edge of the FOV, no rotation is applied.
        '''    
        # Find first available image-like data to check FOV border
        test_image = None
        for img in [image_multChannel, sinogram_multChannel, atten_image_multChannel, atten_sino_multChannel, recon1_multChannel, recon2_multChannel]:
            if img is not None:
                test_image = img
                break
        
        # Only rotate if we have data and it doesn't intersect border
        if test_image is not None and IntersectSquareBorder(test_image) == False:
            angle = np.random.randint(0,360)
            
            # Rotate all non-None data
            if sinogram_multChannel is not None:
                sinogram_multChannel = transforms.functional.rotate(sinogram_multChannel, angle, fill=0)
            if image_multChannel is not None:
                image_multChannel = transforms.functional.rotate(image_multChannel, angle, fill=0)
            if atten_sino_multChannel is not None:
                atten_sino_multChannel = transforms.functional.rotate(atten_sino_multChannel, angle, fill=0)
            if atten_image_multChannel is not None:
                atten_image_multChannel = transforms.functional.rotate(atten_image_multChannel, angle, fill=0)
            if recon1_multChannel is not None:
                recon1_multChannel = transforms.functional.rotate(recon1_multChannel, angle, fill=0)
            if recon2_multChannel is not None:
                recon2_multChannel = transforms.functional.rotate(recon2_multChannel, angle, fill=0)

        return sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel

    def VerticalFlipImageImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel):
        if sinogram_multChannel is not None:
            sinogram_multChannel = torch.flip(sinogram_multChannel,dims=(1,)) # Flip sinogram vertically
        if image_multChannel is not None:
            image_multChannel = torch.flip(image_multChannel,dims=(1,)) # Flip image vertically
        if atten_sino_multChannel is not None:
            atten_sino_multChannel = torch.flip(atten_sino_multChannel, dims=(1,)) # Flip atten_sino vertically
        if atten_image_multChannel is not None:
            atten_image_multChannel = torch.flip(atten_image_multChannel, dims=(1,)) # Flip atten_image vertically
        if recon1_multChannel is not None:
            recon1_multChannel = torch.flip(recon1_multChannel, dims=(1,)) # Flip recon1 vertically
        if recon2_multChannel is not None:
            recon2_multChannel = torch.flip(recon2_multChannel, dims=(1,)) # Flip recon2 vertically
        return sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel

    def HorizontalFlipImageImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel):
        if sinogram_multChannel is not None:
            sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(2,)) # Flip sinogram horizontally
        if image_multChannel is not None:
            image_multChannel = torch.flip(image_multChannel, dims=(2,)) # Flip image horizontally
        if atten_sino_multChannel is not None:
            atten_sino_multChannel = torch.flip(atten_sino_multChannel, dims=(2,)) # Flip atten_sino horizontally
        if atten_image_multChannel is not None:
            atten_image_multChannel = torch.flip(atten_image_multChannel, dims=(2,)) # Flip atten_image horizontally
        if recon1_multChannel is not None:
            recon1_multChannel = torch.flip(recon1_multChannel, dims=(2,)) # Flip recon1 horizontally
        if recon2_multChannel is not None:
            recon2_multChannel = torch.flip(recon2_multChannel, dims=(2,)) # Flip recon2 horizontally
        return sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel

    def ChannelFlipImageImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel):
        if sinogram_multChannel is not None:
            sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(0,)) # Flip sinogram channels about center channel
        if image_multChannel is not None:
            image_multChannel = torch.flip(image_multChannel, dims=(0,)) # Flip image channels about center channel
        if atten_sino_multChannel is not None:
            atten_sino_multChannel = torch.flip(atten_sino_multChannel, dims=(0,)) # Flip atten_sino channels about center channel
        if atten_image_multChannel is not None:
            atten_image_multChannel = torch.flip(atten_image_multChannel, dims=(0,)) # Flip atten_image channels about center channel
        if recon1_multChannel is not None:
            recon1_multChannel = torch.flip(recon1_multChannel, dims=(0,)) # Flip recon1 channels about center channel
        if recon2_multChannel is not None:
            recon2_multChannel = torch.flip(recon2_multChannel, dims=(0,)) # Flip recon2 channels about center channel
        return sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel

    sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel = RandRotateImageImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel)           # Always rotates image by a random angle
    if np.random.choice([True, False]): # Half of the time, flips the image vertically
        sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel = VerticalFlipImageImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel)
    if np.random.choice([True, False]): # Half of the time, flips the image horizontally
        sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel = HorizontalFlipImageImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel)
    if flip_channels==True and np.random.choice([True, False]): # Half of the time, flips the sinogram channels about the center channel
        sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel = ChannelFlipImageImage(sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel)

    return sinogram_multChannel, image_multChannel, atten_sino_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel


def IntersectSquareBorder(image):
    '''
    Function for determining whether the image intersects the edge of the square FOV. If it does not, then the image
    is fully specified by the sinogram and data augmentation can be performed. If the image does
    intersect the edge of the image then some of it may be cropped outside the FOV. In this case,
    augmentation via rotation should not be performed as the rotated image may not be fully described by the sinogram.
    Looks at all channels in the image.
    '''
    max_idx = image.shape[1]-1
    margin_sum = torch.sum(image[:,0,:]).item() + torch.sum(image[:,max_idx,:]).item() \
                +torch.sum(image[:,:,0]).item() + torch.sum(image[:,:,max_idx]).item()
    return_value = False if margin_sum == 0 else True
    return return_value