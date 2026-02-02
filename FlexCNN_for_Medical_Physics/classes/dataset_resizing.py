import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


def vertical_crop_pad_sino(act_sino_tensor=None, atten_sino_tensor=None, target_height=None):
    """
    Vertically crop/pad sinograms to target_height, optimized for numpy domain (before torch conversion).
    Processes stacked sinograms together when both are present for efficiency.
    
    Args:
        act_sino_tensor: Activity sinogram as numpy array [H, W] or None
        atten_sino_tensor: Attenuation sinogram as numpy array [H, W] or None
        target_height: Desired vertical height. If None or matches current height, returns unchanged.
    
    Returns:
        Tuple (act_sino_resized, atten_sino_resized) with None values preserved
    """
    # Early return if no target specified
    if target_height is None:
        return (act_sino_tensor, atten_sino_tensor)
    
    # Determine current height
    if act_sino_tensor is not None:
        current_height = act_sino_tensor.shape[0]
    elif atten_sino_tensor is not None:
        current_height = atten_sino_tensor.shape[0]
    else:
        return (None, None)
    
    # Early return if already at target height
    if target_height == current_height:
        return (act_sino_tensor, atten_sino_tensor)
    
    def _process_single(sino_array):
        """Helper to apply vertical crop/pad to a single sinogram."""
        H, W = sino_array.shape
        if H > target_height:
            # Center crop
            top = (H - target_height) // 2
            return sino_array[top:top + target_height, :]
        elif H < target_height:
            # Center pad with replication
            pad_total = target_height - H
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            # Replicate edges for padding
            top_pad = np.repeat(sino_array[0:1, :], pad_top, axis=0)
            bottom_pad = np.repeat(sino_array[-1:, :], pad_bottom, axis=0)
            return np.vstack([top_pad, sino_array, bottom_pad])
        else:
            return sino_array
    
    # Process sinograms
    if act_sino_tensor is not None and atten_sino_tensor is not None:
        # Stack for batch processing
        stacked = np.stack([act_sino_tensor, atten_sino_tensor], axis=0)  # [2, H, W]
        # Apply crop/pad to each
        processed_stacked = np.array([_process_single(stacked[0]), _process_single(stacked[1])])
        return (processed_stacked[0], processed_stacked[1])
    elif act_sino_tensor is not None:
        return (_process_single(act_sino_tensor), None)
    elif atten_sino_tensor is not None:
        return (None, _process_single(atten_sino_tensor))
    else:
        return (None, None)


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

def crop_pad_sino(
    sino_multChannel,
    atten_sino_multChannel,
    vert_size,
    target_width,
    pool_size=2,
    pad_type='sinogram'
):
    """
    Crop vertically, pool horizontally (if pool_size > 1), then pad horizontally to target_width.
    Applies identical transforms to both sino and atten_sino with efficient stacked processing when both present.
    
    Operation sequence:
    1) Vertical crop/pad to vert_size (if needed)
    2) Horizontal pooling (only if pool_size > 1)
    3) Horizontal padding to target_width

    Args:
        sino_multChannel (torch.Tensor): Input tensor of shape [C, H, W] or None
        atten_sino_multChannel (torch.Tensor): Attenuation sinogram tensor of shape [C, H, W] or None
        vert_size (int): Desired vertical size after cropping/padding
        target_width (int): Desired horizontal size after pooling and padding
        pool_size (int): Number of adjacent angular bins to average (horizontal pooling). If 1, skips pooling.
        pad_type (str): 'sinogram' (mirror vertical axis) or 'zeros' for padding

    Returns:
        Tuple (sino_multChannel, atten_sino_multChannel)
    """

    def _process_sinogram_stack(sino_stack):
        """
        Helper to apply all transforms to sinogram stack [B, C, H, W] where B=1 or B=2.
        Returns processed stack with same batch dimension.
        """
        B, C, H, W = sino_stack.shape
        
        # Step 1: Vertical crop/pad to vert_size
        if H != vert_size:
            if H > vert_size:
                # Center crop
                top = (H - vert_size) // 2
                sino_stack = sino_stack[:, :, top:top + vert_size, :]
            else:
                # Center pad
                pad_total = vert_size - H
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top
                sino_stack = F.pad(sino_stack, (0, 0, pad_top, pad_bottom))
        
        # Step 2: Horizontal pooling (conditional on pool_size > 1)
        if pool_size > 1:
            # Pad horizontally to make width divisible by pool_size
            W_curr = sino_stack.shape[3]
            pad_needed = (pool_size - (W_curr % pool_size)) % pool_size
            
            if pad_needed > 0:
                # Apply replication padding on right side
                sino_stack = F.pad(sino_stack, (0, pad_needed), mode='replicate')
            
            # Apply horizontal-only average pooling (no vertical pooling)
            B, C, H, W = sino_stack.shape
            assert W % pool_size == 0, f"Width {W} must be divisible by pool_size {pool_size}"
            sino_stack = F.avg_pool2d(sino_stack, kernel_size=(1, pool_size), stride=(1, pool_size))
        
        # Step 3: Horizontal padding to target_width
        curr_w = sino_stack.shape[3]
        if curr_w != target_width:
            if curr_w > target_width:
                raise ValueError(f"Width after pooling ({curr_w}) exceeds target_width ({target_width}). Check pool_size and target_width parameters.")
            else:
                # Pad to target_width
                pad_total = target_width - curr_w
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                
                # Build padding based on pad_type
                if pad_type == 'zeros':
                    # Zero padding
                    sino_stack = F.pad(sino_stack, (pad_left, pad_right), mode='constant', value=0)
                else:
                    # Sinogram-style padding: take columns from opposite side and flip vertically
                    # Left padding: take rightmost pad_left columns, flip vertically
                    left_pad = sino_stack[:, :, :, -pad_left:].flip(dims=[2])  # [B, C, H, pad_left]
                    
                    # Right padding: take leftmost pad_right columns, flip vertically
                    right_pad = sino_stack[:, :, :, :pad_right].flip(dims=[2])  # [B, C, H, pad_right]
                    
                    # Concatenate: left_pad | sino_stack | right_pad
                    sino_stack = torch.cat([left_pad, sino_stack, right_pad], dim=3)
        
        return sino_stack
    
    # Determine stacking requirement
    has_act = sino_multChannel is not None
    has_atten = atten_sino_multChannel is not None
    
    if has_act and has_atten:
        # Stack both sinograms for efficient batch processing
        stacked = torch.stack([sino_multChannel, atten_sino_multChannel], dim=0)  # [2, C, H, W]
        processed_stack = _process_sinogram_stack(stacked)  # [2, C, H, W]
        sino_multChannel, atten_sino_multChannel = torch.unbind(processed_stack, dim=0)
    elif has_act:
        # Process activity only
        stacked = sino_multChannel.unsqueeze(0)  # [1, C, H, W]
        processed_stack = _process_sinogram_stack(stacked)  # [1, C, H, W]
        sino_multChannel = processed_stack.squeeze(0)  # [C, H, W]
    elif has_atten:
        # Process attenuation only
        stacked = atten_sino_multChannel.unsqueeze(0)  # [1, C, H, W]
        processed_stack = _process_sinogram_stack(stacked)  # [1, C, H, W]
        atten_sino_multChannel = processed_stack.squeeze(0)  # [C, H, W]
    # else: both None, return as-is
    
    return sino_multChannel, atten_sino_multChannel


def resize_sino_data(
    sino_multChannel,
    atten_sino_multChannel,
    sino_size,
    resize_sino=True,
    sino_resize_type='crop_pad',
    sino_pad_type='sinogram',
    pool_size=2
):
    """
    Resize or pad sinogram data (activity and attenuation) to target sino_size.

    Args:
        sino_multChannel: Activity sinogram tensor of shape (C, H, W) or None
        atten_sino_multChannel: Attenuation sinogram tensor of shape (C, H, W) or None
        sino_size: Target sinogram size (int for square)
        resize_sino: Boolean flag to enable resizing
        sino_resize_type: 'crop_pad' (default) or 'bilinear'
        sino_pad_type: 'sinogram' (default) or 'zeros'
        pool_size: Pool size for crop_pad_sino

    Returns:
        Tuple (sino_resized, atten_sino_resized)
    """
    if resize_sino==False:
        return sino_multChannel, atten_sino_multChannel

    if sino_resize_type=='bilinear':
        return bilinear_resize_sino(sino_multChannel, atten_sino_multChannel, sino_size)

    return crop_pad_sino(
        sino_multChannel,
        atten_sino_multChannel,
        vert_size=sino_size,
        target_width=sino_size,
        pool_size=pool_size,
        pad_type=sino_pad_type
    )




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
