import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

def resize_sino_data(
    act_sino_multChannel,
    atten_sino_multChannel,
    sino_size,
    resize_sino=True,
    sino_resize_type='pool',
    sino_pad_type='sinogram',
    sino_init_vert_cut=None,
    vert_pool_size=1,
    horiz_pool_size=2,
    bilinear_intermediate_size=None
):
    """
    Resize or pad sinogram data (activity and attenuation) to target sino_size.
    
    Two resize paths:
    1) 'bilinear': Bilinear interpolation to intermediate size (if specified) or final size, then pad
    2) 'pool': Pool (vertical/horizontal), crop/pad vertically, pad horizontally

    Args:
        act_sino_multChannel: Activity sinogram tensor of shape (C, H, W) or None
        atten_sino_multChannel: Attenuation sinogram tensor of shape (C, H, W) or None
        sino_size: Target sinogram size (int for square)
        resize_sino: Boolean flag to enable resizing
        sino_resize_type: 'pool' (default) or 'bilinear'
        sino_pad_type: 'sinogram' (default) or 'zeros' for final horizontal padding
        sino_init_vert_cut: Symmetrically crop sinograms to this height before resizing (universal option).
                            If None, no initial crop applied. Works with both 'pool' and 'bilinear' paths.
        vert_pool_size: Vertical pooling factor for pool path (default 1, no pooling)
        horiz_pool_size: Horizontal pooling factor for pool path (default 2)
        bilinear_intermediate_size: For bilinear path, resize to this size before padding.
                                     If None, resize directly to sino_size (no padding needed)

    Returns:
        Tuple (act_sino_resized, atten_sino_resized)
    """
    if resize_sino==False:
        return act_sino_multChannel, atten_sino_multChannel

    # Step 0: Apply initial vertical crop (universal, applies before any resizing method)
    act_sino_multChannel, atten_sino_multChannel = _vertical_crop_sino(
        act_sino_multChannel, atten_sino_multChannel, sino_init_vert_cut
    )

    # Step 1: Resize using selected method
    if sino_resize_type=='bilinear':
        # Determine resize target
        resize_target = bilinear_intermediate_size if bilinear_intermediate_size is not None else sino_size
        
        # Bilinear resize to intermediate or target size
        act_sino_processed, atten_sino_processed = bilinear_resize_sino(
            act_sino_multChannel, atten_sino_multChannel, resize_target
        )
    elif sino_resize_type=='pool':
        # Pool path: apply vertical and horizontal pooling
        act_sino_processed, atten_sino_processed = pool_sino(
            act_sino_multChannel,
            atten_sino_multChannel,
            vert_pool_size=vert_pool_size,
            horiz_pool_size=horiz_pool_size
        )
    else:
        raise ValueError(f"sino_resize_type must be 'bilinear' or 'pool', got '{sino_resize_type}'")
    
    # Step 2: Crop/pad to exact target dimensions (for both paths)
    return _crop_pad_sino_to_target(
        act_sino_processed,
        atten_sino_processed,
        target_height=sino_size,
        target_width=sino_size,
        pad_type=sino_pad_type
    )

def _vertical_crop_sino(act_sino, atten_sino, target_height):
    """
    Symmetrically crop sinograms vertically to target height.
    Applies identical transforms to both with efficient stacked processing when both present.
    
    Args:
        act_sino: Activity sinogram tensor of shape (C, H, W) or None
        atten_sino: Attenuation sinogram tensor of shape (C, H, W) or None
        target_height: Final height after cropping. If None, no crop applied.
    
    Returns:
        Tuple (act_sino_cropped, atten_sino_cropped)
    """
    if target_height is None:
        return act_sino, atten_sino
    
    def _crop_sinogram_stack(sino_stack):
        """Helper to apply vertical crop to sinogram stack [B, C, H, W]."""
        B, C, H, W = sino_stack.shape
        
        if H <= target_height:
            return sino_stack  # Smaller or equal, don't crop
        
        crop_total = H - target_height
        crop_top = crop_total // 2
        crop_bottom = crop_total - crop_top
        
        return sino_stack[:, :, crop_top:H - crop_bottom, :]
    
    # Use stacking for efficiency when both sinograms present and have matching channels
    has_act = act_sino is not None
    has_atten = atten_sino is not None
    
    if has_act and has_atten:
        # Only stack if channel dimensions match; otherwise process separately
        if act_sino.shape[0] == atten_sino.shape[0]:
            stacked = torch.stack([act_sino, atten_sino], dim=0)  # [2, C, H, W]
            cropped_stack = _crop_sinogram_stack(stacked)  # [2, C, H, W]
            act_sino, atten_sino = torch.unbind(cropped_stack, dim=0)
        else:
            # Different channel counts (e.g., CONCAT network): process separately
            stacked_act = act_sino.unsqueeze(0)  # [1, C, H, W]
            cropped_act = _crop_sinogram_stack(stacked_act)  # [1, C, H, W]
            act_sino = cropped_act.squeeze(0)  # [C, H, W]
            
            stacked_atten = atten_sino.unsqueeze(0)  # [1, C, H, W]
            cropped_atten = _crop_sinogram_stack(stacked_atten)  # [1, C, H, W]
            atten_sino = cropped_atten.squeeze(0)  # [C, H, W]
    elif has_act:
        stacked = act_sino.unsqueeze(0)  # [1, C, H, W]
        cropped_stack = _crop_sinogram_stack(stacked)  # [1, C, H, W]
        act_sino = cropped_stack.squeeze(0)  # [C, H, W]
    elif has_atten:
        stacked = atten_sino.unsqueeze(0)  # [1, C, H, W]
        cropped_stack = _crop_sinogram_stack(stacked)  # [1, C, H, W]
        atten_sino = cropped_stack.squeeze(0)  # [C, H, W]
    
    return act_sino, atten_sino

def bilinear_resize_sino(act_sino_multChannel, atten_sino_multChannel, sino_size):
    """
    Resize sinogram data (activity and attenuation) using bilinear interpolation.
    
    Args:
        act_sino_multChannel: Activity sinogram tensor of shape (C, H, W) or None
        atten_sino_multChannel: Attenuation sinogram tensor of shape (C, H, W) or None
        sino_size: Target sinogram size (int for square)
    
    Returns:
        Tuple (act_sino_resized, atten_sino_resized)
    """
    resize_op = transforms.Resize(size=(sino_size, sino_size), antialias=True)
    act_sino_resized = resize_op(act_sino_multChannel) if act_sino_multChannel is not None else None
    atten_sino_resized = resize_op(atten_sino_multChannel) if atten_sino_multChannel is not None else None
    return act_sino_resized, atten_sino_resized

def pool_sino(
    act_sino_multChannel,
    atten_sino_multChannel,
    vert_pool_size=1,
    horiz_pool_size=2
):
    """
    Apply vertical and/or horizontal pooling to sinograms.
    Applies identical transforms to both sino and atten_sino with efficient stacked processing when both present.
    
    Operation sequence:
    1) Vertical pooling (if vert_pool_size > 1) - pads with zeros to make height divisible
    2) Horizontal pooling (if horiz_pool_size > 1) - pads with replication to make width divisible

    Args:
        act_sino_multChannel (torch.Tensor): Activity sinogram tensor of shape [C, H, W] or None
        atten_sino_multChannel (torch.Tensor): Attenuation sinogram tensor of shape [C, H, W] or None
        vert_pool_size (int): Vertical pooling factor. If 1, skips vertical pooling.
        horiz_pool_size (int): Horizontal pooling factor. If 1, skips horizontal pooling.

    Returns:
        Tuple (act_sino_pooled, atten_sino_pooled)
    """

    def _process_sinogram_stack(sino_stack):
        """
        Helper to apply pooling transforms to sinogram stack [B, C, H, W] where B=1 or B=2.
        Returns processed stack with same batch dimension.
        """
        B, C, H, W = sino_stack.shape
        
        # Step 1: Vertical pooling (if vert_pool_size > 1)
        if vert_pool_size > 1:
            # Pad vertically with zeros to make height divisible by vert_pool_size
            H_curr = sino_stack.shape[2]
            pad_needed = (vert_pool_size - (H_curr % vert_pool_size)) % vert_pool_size
            
            if pad_needed > 0:
                # Apply zero padding on bottom
                sino_stack = F.pad(sino_stack, (0, 0, 0, pad_needed), mode='constant', value=0)
            
            # Apply vertical-only average pooling
            B, C, H, W = sino_stack.shape
            assert H % vert_pool_size == 0, f"Height {H} must be divisible by vert_pool_size {vert_pool_size}"
            sino_stack = F.avg_pool2d(sino_stack, kernel_size=(vert_pool_size, 1), stride=(vert_pool_size, 1))
        
        # Step 2: Horizontal pooling (if horiz_pool_size > 1)
        if horiz_pool_size > 1:
            # Pad horizontally with replication to make width divisible by horiz_pool_size
            W_curr = sino_stack.shape[3]
            pad_needed = (horiz_pool_size - (W_curr % horiz_pool_size)) % horiz_pool_size
            
            if pad_needed > 0:
                # Apply replication padding on right side
                sino_stack = F.pad(sino_stack, (0, pad_needed, 0, 0), mode='replicate')
            
            # Apply horizontal-only average pooling
            B, C, H, W = sino_stack.shape
            assert W % horiz_pool_size == 0, f"Width {W} must be divisible by horiz_pool_size {horiz_pool_size}"
            sino_stack = F.avg_pool2d(sino_stack, kernel_size=(1, horiz_pool_size), stride=(1, horiz_pool_size))
        
        return sino_stack
    
    # Determine stacking requirement
    has_act = act_sino_multChannel is not None
    has_atten = atten_sino_multChannel is not None
    
    if has_act and has_atten:
        # Only stack if channel dimensions match; otherwise process separately
        if act_sino_multChannel.shape[0] == atten_sino_multChannel.shape[0]:
            # Stack both sinograms for efficient batch processing
            stacked = torch.stack([act_sino_multChannel, atten_sino_multChannel], dim=0)  # [2, C, H, W]
            processed_stack = _process_sinogram_stack(stacked)  # [2, C, H, W]
            act_sino_multChannel, atten_sino_multChannel = torch.unbind(processed_stack, dim=0)
        else:
            # Different channel counts (e.g., CONCAT network): process separately
            stacked_act = act_sino_multChannel.unsqueeze(0)  # [1, C, H, W]
            processed_act = _process_sinogram_stack(stacked_act)  # [1, C, H, W]
            act_sino_multChannel = processed_act.squeeze(0)  # [C, H, W]
            
            stacked_atten = atten_sino_multChannel.unsqueeze(0)  # [1, C, H, W]
            processed_atten = _process_sinogram_stack(stacked_atten)  # [1, C, H, W]
            atten_sino_multChannel = processed_atten.squeeze(0)  # [C, H, W]
    elif has_act:
        # Process activity only
        stacked = act_sino_multChannel.unsqueeze(0)  # [1, C, H, W]
        processed_stack = _process_sinogram_stack(stacked)  # [1, C, H, W]
        act_sino_multChannel = processed_stack.squeeze(0)  # [C, H, W]
    elif has_atten:
        # Process attenuation only
        stacked = atten_sino_multChannel.unsqueeze(0)  # [1, C, H, W]
        processed_stack = _process_sinogram_stack(stacked)  # [1, C, H, W]
        atten_sino_multChannel = processed_stack.squeeze(0)  # [C, H, W]
    # else: both None, return as-is
    
    return act_sino_multChannel, atten_sino_multChannel

def _crop_pad_sino_to_target(act_sino_multChannel, atten_sino_multChannel, target_height, target_width, pad_type='sinogram'):
    """
    Crop and/or pad sinograms to exact target dimensions. Used by both bilinear and pool resize paths.
    Crops when dimension is too large, pads when too small.
    Vertical padding/cropping: always zero padding, center crop
    Horizontal padding: zero or sinogram-style (mirrored flip)
    
    Args:
        act_sino_multChannel: Activity sinogram tensor of shape (C, H, W) or None
        atten_sino_multChannel: Attenuation sinogram tensor of shape (C, H, W) or None
        target_height: Target height
        target_width: Target width
        pad_type: 'sinogram' or 'zeros' for horizontal padding
    
    Returns:
        Tuple (act_sino_adjusted, atten_sino_adjusted)
    """
    def _adjust_dimensions(sino_stack):
        """Crop/pad a stacked tensor [B, C, H, W] to target dimensions."""
        B, C, H, W = sino_stack.shape
        
        # Vertical adjustment (crop or pad with zeros)
        if H > target_height:
            # Center crop
            top = (H - target_height) // 2
            sino_stack = sino_stack[:, :, top:top + target_height, :]
        elif H < target_height:
            # Center pad with zeros
            pad_total = target_height - H
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            sino_stack = F.pad(sino_stack, (0, 0, pad_top, pad_bottom), mode='constant', value=0)
        
        # Horizontal adjustment (crop or pad)
        W_curr = sino_stack.shape[3]
        if W_curr > target_width:
            # Center crop
            left = (W_curr - target_width) // 2
            sino_stack = sino_stack[:, :, :, left:left + target_width]
        elif W_curr < target_width:
            # Pad to target_width
            pad_total = target_width - W_curr
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            
            if pad_type == 'zeros':
                sino_stack = F.pad(sino_stack, (pad_left, pad_right, 0, 0), mode='constant', value=0)
            else:
                # Sinogram-style padding: mirror and flip vertically
                left_pad = sino_stack[:, :, :, -pad_left:].flip(dims=[2]) if pad_left > 0 else None
                right_pad = sino_stack[:, :, :, :pad_right].flip(dims=[2]) if pad_right > 0 else None
                
                parts = [p for p in [left_pad, sino_stack, right_pad] if p is not None]
                sino_stack = torch.cat(parts, dim=3)
        
        return sino_stack
    
    # Process sinograms separately (they have different channel counts)
    has_act = act_sino_multChannel is not None
    has_atten = atten_sino_multChannel is not None
    
    if has_act and has_atten:
        stacked_act = act_sino_multChannel.unsqueeze(0)
        adjusted_act = _adjust_dimensions(stacked_act)
        act_sino_multChannel = adjusted_act.squeeze(0)
        
        stacked_atten = atten_sino_multChannel.unsqueeze(0)
        adjusted_atten = _adjust_dimensions(stacked_atten)
        atten_sino_multChannel = adjusted_atten.squeeze(0)
    elif has_act:
        stacked = act_sino_multChannel.unsqueeze(0)
        adjusted_stack = _adjust_dimensions(stacked)
        act_sino_multChannel = adjusted_stack.squeeze(0)
    elif has_atten:
        stacked = atten_sino_multChannel.unsqueeze(0)
        adjusted_stack = _adjust_dimensions(stacked)
        atten_sino_multChannel = adjusted_stack.squeeze(0)
    
    return act_sino_multChannel, atten_sino_multChannel

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
