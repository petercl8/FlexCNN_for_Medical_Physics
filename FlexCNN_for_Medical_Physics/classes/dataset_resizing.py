import torch
from torchvision import transforms
import torch.nn.functional as F


def ResizeImageData(act_map_multChannel, recon1_multChannel, recon2_multChannel, image_size, resize_image=True, image_pad_type='none'):
    """
    Resize or pad image data (activity map and reconstructions) to target image_size.

    Args:
        act_map_multChannel: Activity map tensor of shape (C, H, W)
        recon1_multChannel: Reconstruction 1 tensor or None
        recon2_multChannel: Reconstruction 2 tensor or None
        image_size: Target image size (int for square)
        resize_image: Boolean flag to enable resizing when image_pad_type='none'
        image_pad_type: 'none' (default) to resize, 'zeros' to pad with zeros without resizing

    Returns:
        Tuple of (act_map_processed, recon1_processed, recon2_processed)
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
        return act_map_multChannel, recon1_multChannel, recon2_multChannel

    else:
        if image_pad_type == 'zeros':
            act_map_out = _pad_to_size(act_map_multChannel)
            recon1_out = _pad_to_size(recon1_multChannel)
            recon2_out = _pad_to_size(recon2_multChannel)
        else:
            resize_op = transforms.Resize(size=(image_size, image_size), antialias=True)
            act_map_out = resize_op(act_map_multChannel)
            recon1_out = resize_op(recon1_multChannel) if recon1_multChannel is not None else None
            recon2_out = resize_op(recon2_multChannel) if recon2_multChannel is not None else None

    return act_map_out, recon1_out, recon2_out


def CropPadSino(
    sino_multChannel,
    vert_size,
    target_width,
    pool_size=2,
    pad_type='sinogram'
):
    """
    Crop vertically, pad horizontally, then perform horizontal average pooling
    to reduce angular resolution, producing a sinogram of desired target_width.

    Args:
        sino_multChannel (torch.Tensor): Input tensor of shape [C, H, W]
        vert_size (int): Desired vertical size after cropping/padding
        target_width (int): Desired horizontal size after pooling
        pool_size (int): Number of adjacent angular bins to average (horizontal pooling)
        pad_type (str): 'sinogram' (mirror vertical axis) or 'zeros' for padding

    Returns:
        torch.Tensor: Processed sinogram tensor of shape [C, vert_size, target_width]
    """

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

    C, H, W = sino_multChannel.shape

    # Vertical crop/pad to vert_size
    if H > vert_size:
        top = (H - vert_size) // 2
        sino_multChannel = sino_multChannel[:, top:top + vert_size, :]
    elif H < vert_size:
        pad_total = vert_size - H
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        sino_multChannel = F.pad(sino_multChannel, (0, 0, pad_top, pad_bottom))

    # Pre-pad horizontally to reach target_width * pool_size
    pre_pool_width = target_width * pool_size
    pad_needed = max(0, pre_pool_width - W)

    left_pad = _build_pad(sino_multChannel, pad_needed // 2, side='left', pad_type=pad_type)
    right_pad = _build_pad(sino_multChannel, pad_needed - pad_needed // 2, side='right', pad_type=pad_type)

    if left_pad is not None and right_pad is not None:
        sino_multChannel = torch.cat([left_pad, sino_multChannel, right_pad], dim=2)
    elif left_pad is not None:
        sino_multChannel = torch.cat([left_pad, sino_multChannel], dim=2)
    elif right_pad is not None:
        sino_multChannel = torch.cat([sino_multChannel, right_pad], dim=2)

    # Average-pool adjacent bins horizontally
    C, H, W = sino_multChannel.shape
    assert W % pool_size == 0, "Pre-padded width must be divisible by pool_size"
    sino_multChannel = sino_multChannel.view(C, H, W // pool_size, pool_size).mean(dim=3)

    # Final horizontal adjust to target_width
    curr_w = sino_multChannel.shape[2]
    if curr_w > target_width:
        start = (curr_w - target_width) // 2
        sino_multChannel = sino_multChannel[:, :, start:start + target_width]
    elif curr_w < target_width:
        pad_total = target_width - curr_w
        left_pad = _build_pad(sino_multChannel, pad_total // 2, side='left', pad_type=pad_type)
        right_pad = _build_pad(sino_multChannel, pad_total - pad_total // 2, side='right', pad_type=pad_type)
        if left_pad is not None and right_pad is not None:
            sino_multChannel = torch.cat([left_pad, sino_multChannel, right_pad], dim=2)
        elif left_pad is not None:
            sino_multChannel = torch.cat([left_pad, sino_multChannel], dim=2)
        elif right_pad is not None:
            sino_multChannel = torch.cat([sino_multChannel, right_pad], dim=2)

    return sino_multChannel