__all__=['crop_single_image_by_size',
         'crop_single_image_by_factor',
         'crop_image_tensor_by_factor',
         'crop_image_tensor_with_corner']     

def crop_single_image_by_size(image, crop_size=-1):
    '''
    Function to crop a single image to a square shape, with even margins around the edges.

    image:       Input image tensor of shape [height, width]
    crop_size:   Edge size of (square) image to keep. The edges are discarded.
    '''
    x_size = image.shape[1]

    margin_low = int((x_size-crop_size)/2.0)  # (90-71)/2 = 19/2 = 9.5 -->9
    margin_high = x_size-crop_size-margin_low # 90-71-9 = 10

    pix_min = 0 + margin_low
    pix_max = x_size - margin_high

    image = image[pix_min : pix_max , pix_min : pix_max]

    return image

def crop_single_image_by_factor(image, crop_factor=1):
    '''
    Function to crop a single image for a factor, with even margins around the edges.

    image:       Input image tensor of shape [height, width]
    crop_factor: Fraction of image to keep. The image is trimmed so the edges are discarded.
    '''
    x_size = image.shape[1]
    y_size = image.shape[0]

    x_margin = int(x_size*(1-crop_factor)/2)
    y_margin = int(y_size*(1-crop_factor)/2)

    x_min = 0 + x_margin
    x_max = x_size - x_margin
    y_min = 0 + y_margin
    y_max = y_size - y_margin

    return image[y_min:y_max , x_min:x_max]


def crop_image_tensor_with_corner(batch, crop_size, corner=(0,0)):
    '''
    Function which returns a smaller, cropped version of a tensor (multiple images)

    batch:       a batch of images with dimensions: (num_images, channel, y_dimension, x_dimension)
    corner:      upper-left corner of window
    crop_size:   size of cropping window (int)
    '''

    y_min = corner[0]
    y_max = corner[0]+crop_size
    x_min = corner[1]
    x_max = corner[1]+crop_size

    return batch[:, :, y_min:y_max , x_min:x_max ]


def crop_image_tensor_by_factor(image_tensor, crop_factor=1):
    '''
    Function to crop an image tensor, with even margins around the edges.

    image_tensor:   Input image tensor of shape [image number, channel, height, width]
    crop_factor:    Fraction of image to keep. The images are trimmed so the edges are discarded.
    '''
    x_size = image_tensor.shape[3]
    y_size = image_tensor.shape[2]

    x_margin = int(x_size*(1-crop_factor)/2)
    y_margin = int(y_size*(1-crop_factor)/2)

    x_min = 0 + x_margin
    x_max = x_size - x_margin
    y_min = 0 + y_margin
    y_max = y_size - y_margin

    return image_tensor[:,:, y_min:y_max , x_min:x_max ]