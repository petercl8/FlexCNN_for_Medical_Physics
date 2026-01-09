import torch
from torchvision import transforms
import torch.nn.functional as F


def bilinear_resize_sino(sino_multChannel, atten_sino_multChannel, sino_size):
    """
    Resize sinogram data (activity and attenuation) using bilinear interpolation.
    
    Args:
        sino_multChannel: Activity sinogram tensor of shape (C, H, W) or None
        atten_sino_multChannel: Attenuation sinogram tensor of shape (C, H, W) or None
        sino_size: Target sinogram size (int for square)
    
    Returns:
        Tuple (sino_resized, atten_sino_resized)
    """
    resize_op = transforms.Resize(size=(sino_size, sino_size), antialias=True)
    sino_resized = resize_op(sino_multChannel) if sino_multChannel is not None else None
    atten_sino_resized = resize_op(atten_sino_multChannel) if atten_sino_multChannel is not None else None
    return sino_resized, atten_sino_resized


def resize_image_data(image_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel, image_size, resize_image=True, image_pad_type='none'):
    """
    Resize or pad image data (activity map, attenuation image, and reconstructions) to target image_size.

    Args:
        act_map_multChannel: Activity map tensor of shape (C, H, W) or None
        atten_image_multChannel: Attenuation image tensor of shape (C, H, W) or None
        recon1_multChannel: Reconstruction 1 tensor of shape (C, H, W) or None
        recon2_multChannel: Reconstruction 2 tensor of shape (C, H, W) or None
        image_size: Target image size (int for square)
        resize_image: Boolean flag to enable resizing when image_pad_type='none'
        image_pad_type: 'none' (default) to resize w/ bilinear interpolation to correct size, 'zeros' to pad with zeros without resizing

    Returns:
        Tuple of (image_processed, atten_image_processed, recon1_processed, recon2_processed)
    """

    def _pad_to_size(tensor):
        if tensor is None:
            return None
        C, H, W = tensor.shape
        pad_h = max(0, image_size - H)
        pad_w = max(0, image_size - W)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))
        return tensor

    if resize_image==False:
        return image_multChannel, atten_image_multChannel, recon1_multChannel, recon2_multChannel

    else:
        if image_pad_type == 'zeros':
            image_out = _pad_to_size(image_multChannel)
            atten_image_out = _pad_to_size(atten_image_multChannel)
            recon1_out = _pad_to_size(recon1_multChannel)
            recon2_out = _pad_to_size(recon2_multChannel)
        else:
            resize_op = transforms.Resize(size=(image_size, image_size), antialias=True)
            image_out = resize_op(image_multChannel) if image_multChannel is not None else None
            atten_image_out = resize_op(atten_image_multChannel) if atten_image_multChannel is not None else None
            recon1_out = resize_op(recon1_multChannel) if recon1_multChannel is not None else None
            recon2_out = resize_op(recon2_multChannel) if recon2_multChannel is not None else None

    return image_out, atten_image_out, recon1_out, recon2_out


def crop_pad_sino(
    sino_multChannel,
    atten_sino_multChannel,
    vert_size,
    target_width,
    pool_size=2,
    pad_type='sinogram'
):
    """
    Crop vertically, pad horizontally, then perform horizontal average pooling
    to reduce angular resolution, producing a sinogram of desired target_width.
    Applies identical transforms to both sino and atten_sino.

    Args:
        sino_multChannel (torch.Tensor): Input tensor of shape [C, H, W] or None
        atten_sino_multChannel (torch.Tensor): Attenuation sinogram tensor of shape [C, H, W] or None
        vert_size (int): Desired vertical size after cropping/padding
        target_width (int): Desired horizontal size after pooling
        pool_size (int): Number of adjacent angular bins to average (horizontal pooling)
        pad_type (str): 'sinogram' (mirror vertical axis) or 'zeros' for padding

    Returns:
        Tuple (sino_multChannel, atten_sino_multChannel)
    """

    def _process_single_sino(sino_tensor):
        """Helper to apply all transforms to a single sinogram tensor."""
        C, H, W = sino_tensor.shape

        # Vertical crop/pad to vert_size
        if H > vert_size:
            top = (H - vert_size) // 2
            sino_tensor = sino_tensor[:, top:top + vert_size, :]
        elif H < vert_size:
            pad_total = vert_size - H
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            sino_tensor = F.pad(sino_tensor, (0, 0, pad_top, pad_bottom))

        # Pre-pad horizontally to reach target_width * pool_size
        pre_pool_width = target_width * pool_size
        W_curr = sino_tensor.shape[2]
        pad_needed = max(0, pre_pool_width - W_curr)

        left_pad = _build_pad(sino_tensor, pad_needed // 2, side='left', pad_type=pad_type)
        right_pad = _build_pad(sino_tensor, pad_needed - pad_needed // 2, side='right', pad_type=pad_type)

        if left_pad is not None and right_pad is not None:
            sino_tensor = torch.cat([left_pad, sino_tensor, right_pad], dim=2)
        elif left_pad is not None:
            sino_tensor = torch.cat([left_pad, sino_tensor], dim=2)
        elif right_pad is not None:
            sino_tensor = torch.cat([sino_tensor, right_pad], dim=2)

        # Average-pool adjacent bins horizontally
        C, H, W = sino_tensor.shape
        assert W % pool_size == 0, "Pre-padded width must be divisible by pool_size"
        sino_tensor = sino_tensor.view(C, H, W // pool_size, pool_size).mean(dim=3)

        # Final horizontal adjust to target_width
        curr_w = sino_tensor.shape[2]
        if curr_w > target_width:
            start = (curr_w - target_width) // 2
            sino_tensor = sino_tensor[:, :, start:start + target_width]
        elif curr_w < target_width:
            pad_total = target_width - curr_w
            left_pad = _build_pad(sino_tensor, pad_total // 2, side='left', pad_type=pad_type)
            right_pad = _build_pad(sino_tensor, pad_total - pad_total // 2, side='right', pad_type=pad_type)
            if left_pad is not None and right_pad is not None:
                sino_tensor = torch.cat([left_pad, sino_tensor, right_pad], dim=2)
            elif left_pad is not None:
                sino_tensor = torch.cat([left_pad, sino_tensor], dim=2)
            elif right_pad is not None:
                sino_tensor = torch.cat([sino_tensor, right_pad], dim=2)

        return sino_tensor

    def _build_pad(source, pad_width, side, pad_type):
        if pad_width <= 0:
            return None
        if pad_type == 'zeros':
            return torch.zeros(
                (source.shape[0], source.shape[1], pad_width),
                device=source.device,
                dtype=source.dtype
            )
        # Sinogram-style padding: take edge columns and flip vertically
        if side == 'left':
            tile = source[:, :, -pad_width:]
        else:  # right
            tile = source[:, :, :pad_width]
        tile = torch.flip(tile, dims=[1])
        return tile

    # Process activity sinogram if provided
    if sino_multChannel is not None:
        sino_multChannel = _process_single_sino(sino_multChannel)

    # Process attenuation sinogram if provided
    if atten_sino_multChannel is not None:
        atten_sino_multChannel = _process_single_sino(atten_sino_multChannel)

    return sino_multChannel, atten_sino_multChannel