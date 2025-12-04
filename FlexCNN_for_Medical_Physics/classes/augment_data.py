import numpy as np
import torch
import torchvision.transforms as transforms

def AugmentSinoImageData(image_multChannel, sinogram_multChannel, flip_channels=False):
    ## Data Augmentation Functions ##
    def RandRotateSinoImage(image_multChannel, sinogram_multChannel):
        '''
        Function for randomly rotating an image and its sinogram. If the image intersects the edge of the FOV, no rotation is applied.

        image_multChannel:    image to rotate. Shape: (C, H, W)
        sinogram_multChannel: sinogram to rotate. Shape: (C, H, W)
        '''    
        if IntersectSquareBorder(image_multChannel) == False:
            bins = sinogram_multChannel.shape[2]
            bins_shifted = np.random.randint(0, bins)
            angle = int(bins_shifted * 180/bins)

            image_multChannel = transforms.functional.rotate(image_multChannel, angle, fill=0) # Rotate image. Fill in unspecified pixels with zeros.
            sinogram_multChannel = torch.roll(sinogram_multChannel, bins_shifted, dims=(2,)) # Cycle (or 'Roll') sinogram by that angle along dimension 2.
            sinogram_multChannel[:,:, 0:bins_shifted] = torch.flip(sinogram_multChannel[:,:,0:bins_shifted], dims=(1,)) # flip the cycled portion of the sinogram vertically

        return image_multChannel, sinogram_multChannel

    def VerticalFlipSinoImage(image_multChannel, sinogram_multChannel):
        image_multChannel = torch.flip(image_multChannel,dims=(1,)) # Flip image vertically
        sinogram_multChannel = torch.flip(sinogram_multChannel,dims=(1,2)) # Flip sinogram horizontally and vertically
        return image_multChannel, sinogram_multChannel

    def HorizontalFlipSinoImage(image_multChannel, sinogram_multChannel):
        image_multChannel = torch.flip(image_multChannel, dims=(2,)) # Flip image horizontally
        sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(2,)) # Flip sinogram horizontally
        return image_multChannel, sinogram_multChannel

    def ChannelFlipSinoImage(image_multChannel, sinogram_multChannel):
        sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(0,)) # Flip sinogram channels about center channel
        return image_multChannel, sinogram_multChannel

    image_multChannel, sinogram_multChannel = RandRotateSinoImage(image_multChannel, sinogram_multChannel)           # Always rotates image by a random angle
    if np.random.choice([True, False]): # Half of the time, flips the image vertically
        image_multChannel, sinogram_multChannel = VerticalFlipSinoImage(image_multChannel, sinogram_multChannel)
    if np.random.choice([True, False]): # Half of the time, flips the image horizontally
        image_multChannel, sinogram_multChannel = HorizontalFlipSinoImage(image_multChannel, sinogram_multChannel)
    if flip_channels==True and np.random.choice([True, False]): # Half of the time, flips the sinogram channels about the center channel
        image_multChannel, sinogram_multChannel = ChannelFlipSinoImage(image_multChannel, sinogram_multChannel)

    return image_multChannel, sinogram_multChannel


def AugmentImageImageData(image_multChannel, sinogram_multChannel, flip_channels=False):
    ## If one would like to train a network that goes from image to image (e.g. an image denoising network),
    ## this function can be used to augment the paired image data. To keep nomenclature consistent, the sinogram input is still called 'sinogram_multChannel' here.
    ## Data Augmentation Functions ##
    def RandRotateImageImage(image_multChannel, sinogram_multChannel):
        '''
        Function for randomly rotating two paired images. If the image intersects the edge of the FOV, no rotation is applied.
        '''    
        if IntersectSquareBorder(image_multChannel) == False:
            angle = np.random.randint(0,360)
            image_multChannel = transforms.functional.rotate(image_multChannel, angle, fill=0) # Rotate image. Fill in unspecified pixels with zeros.
            sinogram_multChannel = transforms.functional.rotate(sinogram_multChannel, angle, fill=0) # Rotate sinogram. Fill in unspecified pixels with zeros.

        return image_multChannel, sinogram_multChannel

    def VerticalFlipImageImage(image_multChannel, sinogram_multChannel):
        image_multChannel = torch.flip(image_multChannel,dims=(1,)) # Flip image vertically
        sinogram_multChannel = torch.flip(sinogram_multChannel,dims=(1,)) # Flip sinogram vertically
        return image_multChannel, sinogram_multChannel

    def HorizontalFlipImageImage(image_multChannel, sinogram_multChannel):
        image_multChannel = torch.flip(image_multChannel, dims=(2,)) # Flip image horizontally
        sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(2,)) # Flip sinogram horizontally
        return image_multChannel, sinogram_multChannel

    def ChannelFlipImageImage(image_multChannel, sinogram_multChannel):
        image_multChannel = torch.flip(image_multChannel, dims=(0,)) # Flip image channels about center channel
        sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(0,)) # Flip sinogram channels about center channel
        return image_multChannel, sinogram_multChannel

    image_multChannel, sinogram_multChannel = RandRotateImageImage(image_multChannel, sinogram_multChannel)           # Always rotates image by a random angle
    if np.random.choice([True, False]): # Half of the time, flips the image vertically
        image_multChannel, sinogram_multChannel = VerticalFlipImageImage(image_multChannel, sinogram_multChannel)
    if np.random.choice([True, False]): # Half of the time, flips the image horizontally
        image_multChannel, sinogram_multChannel = HorizontalFlipImageImage(image_multChannel, sinogram_multChannel)
    if flip_channels==True and np.random.choice([True, False]): # Half of the time, flips the sinogram channels about the center channel
        image_multChannel, sinogram_multChannel = ChannelFlipImageImage(image_multChannel, sinogram_multChannel)

    return image_multChannel, sinogram_multChannel


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

'''
def IntersectCircularBorder(image):
    
    #Currently unused.
    #Function for determining whether an image itersects a circular boundary inscribed within the square FOV.
    #This function is not currently used.
    
    y_max = image.shape[1]
    x_max = image.shape[2]

    r_max = y_max/2.0
    x_center = (x_max-1)/2.0 # the -1 comes from the fact that the coordinates of a pixel start at 0, not 1
    y_center = (y_max-1)/2.0

    margin_sum = 0
    for y in range(0, y_max):
        for x in range(0, x_max):
            if r_max < ((x-x_center)**2 + (y-y_center)**2)**0.5 :
                margin_sum += torch.sum(image[:,y,x]).item()

    return_value = True if margin_sum == 0 else False
    return return_value
'''
