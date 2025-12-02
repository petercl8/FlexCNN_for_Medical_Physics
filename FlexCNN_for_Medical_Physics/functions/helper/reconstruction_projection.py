import numpy as np
from skimage.transform import iradon, radon # For reconstruction and projection
from torchvision import transforms  # For resizing tensors
from .cropping import crop_single_image_by_factor
import torch.nn as nn
import torch         

def iradon_MLEM(sino_ground, azi_angles=None, max_iter=15, circle=True, crop_factor=2**0.5/2):
    '''
    Function to reconstruct a single PET image from a single sinogram using ML-EM.

    sino_ground:    sinogram (photopeak). This is a numpy array with minimum values of 0. Shape: (H,W)
    azi_angles:     list of azimuthal angles for sinogram. If set to None, angles are assumed to span [0,180)
    max_iter:       Maximum number of iterations for ML-EM algorithm.
    circle:         circle=True: The projection data spans the width (or height) of the activity distribution, and the reconstructed image is circular.
                    circle=False: The projection data (sinograms) spans the corner-to-corner line of the activity distribution, and the reconstructed image is square.
    crop_size:      Size to crop the image to after performing ML-EM. For ML-EM performed on a 90x90 sinogram, the
                    output image will be 90x90. However, it is necessary to crop this to 64x64 to get the same FOV
                    as the dataset. This means the image must cropped by a factor of sqrt(2)/2.
    '''
    if azi_angles==None:
        num_angles = sino_ground.shape[1] # Width
        azi_angles=np.linspace(0, 180, num_angles, endpoint=False)

    ## Create Sensitivity Image ##
    sino_ones = np.ones(sino_ground.shape)
    sens_image = iradon(sino_ones, azi_angles, circle=circle, filter_name=None)

    if circle==False:
        def modify_sens(image, const_factor=0.9, slope=0.03):
            '''
            Modifies an image so that the area in the central FOV remains constant, but values at edges are attenuated.
            image               image to modify
            constant_factor     fraction of the image to leave alone
            slope               increase this to attenuate images at the edges more
            '''
            def shape_piecewise(r, const_value, slope):
                if r <= const_value:
                    return 1
                else:
                    return 1+slope*(r-const_value)

            y_max = image.shape[0]
            x_max = image.shape[1]

            const_dist = const_factor*x_max/2 # radius over which image remains constant

            x_center = (x_max-1)/2.0 # the -1 comes from the fact that the coordinates of a pixel start at 0, not 1
            y_center = (y_max-1)/2.0

            for y in range(0, y_max):
                for x in range(0, x_max):
                    r = ((x-x_center)**2 + (y-y_center)**2)**0.5

                    total_factor = shape_piecewise(r, const_dist, slope) # creates a circular shaped piece-wise
                    #total_factor = shape_piecewise(abs(x-x_center), const_dist, slope) * shape_piecewise(abs(y-y_center), const_dist, slope) # square-shaped piecewise
                    #total_factor = shape_piecewise(abs(y-y_center), const_dist, slope) # vertical only

                    image[y,x] = image[y,x]*total_factor

            return image
        sens_image = modify_sens(sens_image)

    ## Create blank reconstruction ##
    image_recon  = np.ones(sens_image.shape)

    for iter in range(max_iter):

        if circle==True:
            sens_image = sens_image + 0.001 # Guarantees the denominator is >0

        sino_recon = radon(image_recon, azi_angles, circle=circle) #
        sino_recon[sino_recon==0]=1000 # Set a limit on the denominator (next line)
        sino_ratio = sino_ground / (sino_recon) #
        image_ratio = iradon(sino_ratio, azi_angles, circle=circle, filter_name=None) / sens_image
        image_ratio[image_ratio>1.5]=1.5 # Sets limit on backprojected ratio, on how fast image can grow. Threshold and set value should equal each other (good value=1.5)
        image_recon = image_recon * image_ratio
        image_recon[image_recon<0]=0 # Sets floor on image pixels. No need to adjust.

        #footprint = morphology.disk(1)
        #image_recon = opening(image_recon, footprint)

    image_cropped = crop_single_image_by_factor(image_recon, crop_factor=crop_factor)
    #image_cropped = crop_single_image_by_size(image_recon, crop_size=crop_size)

    return image_cropped

def reconstruct(sinogram_tensor, image_size, SI_normalize, SI_fixedScale, recon_type='FBP', circle=True):
    '''
    Function for calculating a reconstructed PET image tensor, given a sinogram_tensor. One image is reconstructed for
    each sinogram in the sinogram_tensor.

    sinogram_tensor:    Tensor of sinograms of size (number of images)x(channels)x(height)x(width).
                        Only the first channel (photopeak) is used for recontruction here.
    image_size:         size of output (images are resized to this shape)
    SI_normalize:       whether to normalize the reconstructed images
    SI_fixedScale:           scaling factor to apply to normalized images
    recon_type:         Can be set to 'MLEM' for maximum-likelihood expectation maximization, or 'FBP' for
                        filtered back-projection.
    circle              circle=True: The projection data spans the width (or height) of the activity distribution, and the reconstructed image is circular.
                        circle=False: The projection data (sinograms) spans the corner-to-corner line of the activity distribution, and the reconstructed image is square.

    Function returns a tensor of reconstructed images. Returned images are resized, and optionally normalized and scaled.
    '''
    photopeak_array = torch.clamp(sinogram_tensor[:,0,:,:], min=0).detach().cpu().numpy()  # Here, we collapse the channel dimension.
    # Note: there really should be no need to clamp the sinogram, as it should contain no negative values, but might as well.

    ## Reconstruct Individual Sinograms ##
    first=True
    for sino in photopeak_array[0:,]:
        if recon_type == 'FBP':
            image = iradon(sino.squeeze(), # Sinogram is now 2D
                        circle=False, # For an unknown reason, circle=False gives better reconstructions here. Maybe due to errors introduced in interpolation.
                        preserve_range=True,
                        filter_name='cosine' # Options: 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'
                        )
        else:
            image = iradon_MLEM(sino, circle=circle)

        ## Morphologic Opening - removes outlier pixels than can cause problems with image normalization
        #footprint = morphology.disk(1)
        #image = opening(image, footprint)

        ## Concatenate Images ##
        image = np.expand_dims(image, axis=0) # Add a dimension to the beginning of the reconstructed image
        if first==True:
            image_array = image
            first=False
        else:
            image_array = np.append(image_array, image, axis=0)

    ## For All Images: create resized/dimensioned Torch tensor ##
    image_array = np.expand_dims(image_array, axis=1)        # Creates channels dimension
    a = torch.from_numpy(image_array)                        # Converts to Torch tensor
    a = torch.clamp(a, min=0)                                # You HAVE to clamp before normalizing or the negative values throw it off.
    a = transforms.Resize(size = (image_size, image_size), antialias=True)(a) # Resize tensor

    ## Normalize Entire Tensor ##
    if SI_normalize:
        batch_size = len(a)
        a = torch.reshape(a,(batch_size, 1, image_size**2)) # Flattens each image
        a = nn.functional.normalize(a, p=1, dim = 2)
        a = torch.reshape(a,(batch_size, 1 , image_size, image_size)) # Reshapes images back into square matrices
        a = SI_fixedScale*a

    # Return the reconstructed images as a tensor on the same device as the sinogram_tensor
    return a.to(sinogram_tensor.device)



def project(image_tensor, sino_size, IS_normalize, SI_fixedScale, circle=False, theta=-1):
    '''
    Perform the forward radon transform to calculate projections from images. Returns an array of sinograms.

    image_tensor:   tensor of PET images
    theta:          numpy array of projection angles. Default is [0,180)
    '''
    image_collapsed = torch.clamp(image_tensor[:,0,:,:], min=0).detach().squeeze().cpu().numpy()

    if theta==-1:
        theta = np.arange(0,180)

    first=True
    for image in image_collapsed[0:,]:
        sino = radon(image,
                    circle=circle,
                    preserve_range=True,
                    theta=theta,
                    )
        sino = np.moveaxis(np.atleast_3d(sino), 2, 0) # Adds a blank axis and moves it to the beginning
        if first==True:
            sino_array=sino
            first=False
        else:
            sino_array = np.append(sino_array, sino, axis=0)

    ## For all sinograms: create resized/dimensioned Torch tensor ##
    sino_array = np.expand_dims(sino_array, axis=1)        # Creates channels dimension
    a = torch.from_numpy(sino_array)                        # Converts to Torch tensor
    a = torch.clamp(a, min=0)                                # You HAVE to clamp before normalizing or the negative values throw it off.
    a = transforms.Resize(size = (sino_size, sino_size), antialias=True)(a) # Resize tensor

    ## Normalize Entire Tensor ##
    if IS_normalize:
        batch_size = len(a)
        a = torch.reshape(a,(batch_size, 1, sino_size**2)) # Flattens each image
        a = nn.functional.normalize(a, p=1, dim = 2)
        a = torch.reshape(a,(batch_size, 1 , sino_size, sino_size)) # Reshapes images back into square matrices
        a = SI_fixedScale*a

    # Return the reconstructed images as a tensor on the same device as the sinogram_tensor
    return a.to(image_tensor.device)
