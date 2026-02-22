import os
import torch
import numpy as np
import sys
import importlib

from FlexCNN_for_Medical_Physics.classes.dataset.dataset_classes import NpArrayDataLoader
from FlexCNN_for_Medical_Physics.classes.generators import Generator_180, Generator_256, Generator_288, Generator_320
from FlexCNN_for_Medical_Physics.functions.helper.model_setup.setup_generators_optimizer import instantiate_dual_generators, load_dual_generator_checkpoints
from FlexCNN_for_Medical_Physics.functions.helper.image_processing.display_images import show_multiple_commonmap_tensors, show_multiple_unmatched_tensors, show_single_unmatched_tensor
from FlexCNN_for_Medical_Physics.functions.helper.model_setup.config_materialize import materialize_config



# Updated: BuildImageSinoTensors returns lists of all images and sinograms (activity, attenuation), and reconstructions if present

def BuildTensors(act_image_array_name, act_sino_array_name, atten_image_array_name, atten_sino_array_name, recon1_array_name, recon2_array_name, config, paths, indexes, device, settings,
                 sino_resize_type='pool', sino_pad_type='zeros', image_pad_type='zeros',
                 sino_init_vert_cut=None, vert_pool_size=1, horiz_pool_size=1, bilinear_intermediate_size=161):
    '''
    Return a dictionary of tensors for activity images/sinograms, attenuation images/sinograms, and reconstructions.
    Each tensor has shape (N, C, H, W), where N is the number of indexes provided (len(indexes)).
    '''
    def load_array(name):
        if name is None:
            return None
        return np.load(os.path.join(paths['data_dirPath'], name), mmap_mode='r')

    act_image_array = load_array(act_image_array_name)
    act_sino_array = load_array(act_sino_array_name)
    atten_image_array = load_array(atten_image_array_name)
    atten_sino_array = load_array(atten_sino_array_name)
    recon1_array = load_array(recon1_array_name)
    recon2_array = load_array(recon2_array_name)

    act_image_tensors = []
    act_sino_tensors = []
    atten_image_tensors = []
    atten_sino_tensors = []
    recon1_tensors = []
    recon2_tensors = []

    for idx in indexes:
        act_data, atten_data, recon_data = NpArrayDataLoader(
            act_sino_array, act_image_array, atten_image_array, atten_sino_array, recon1_array, recon2_array,
            config, settings, augment=(None, False), index=idx, device=device,
            sino_resize_type=sino_resize_type, sino_pad_type=sino_pad_type, image_pad_type=image_pad_type,
            sino_init_vert_cut=sino_init_vert_cut, vert_pool_size=vert_pool_size, horiz_pool_size=horiz_pool_size,
            bilinear_intermediate_size=bilinear_intermediate_size
        )
        act_sino_scaled, act_image_scaled = act_data
        atten_sino_scaled, atten_image_scaled = atten_data
        act_recon1, act_recon2 = recon_data

        if act_image_scaled is not None:
            act_image_tensors.append(act_image_scaled)
        if act_sino_scaled is not None:
            act_sino_tensors.append(act_sino_scaled)
        if atten_image_scaled is not None:
            atten_image_tensors.append(atten_image_scaled)
        if atten_sino_scaled is not None:
            atten_sino_tensors.append(atten_sino_scaled)
        if act_recon1 is not None:
            recon1_tensors.append(act_recon1)
        if act_recon2 is not None:
            recon2_tensors.append(act_recon2)

    return {
        'act_image_tensor': torch.stack(act_image_tensors) if act_image_tensors else None,
        'act_sino_tensor': torch.stack(act_sino_tensors) if act_sino_tensors else None,
        'atten_image_tensor': torch.stack(atten_image_tensors) if atten_image_tensors else None,
        'atten_sino_tensor': torch.stack(atten_sino_tensors) if atten_sino_tensors else None,
        'recon1_tensor': torch.stack(recon1_tensors) if recon1_tensors else None,
        'recon2_tensor': torch.stack(recon2_tensors) if recon2_tensors else None,
    }


# Single-network reconstruction (ACT, ATTEN, CONCAT)
def cnn_reconstruct_single(input_tensor, config, paths, device, checkpoint_name):
    '''
    Single-network reconstruction using specified generator type.
    input_tensor: input for the network (act_sino for ACT, atten_sino for ATTEN, concatenated for CONCAT)
    '''

    # Force reimport of generator classes to ensure fresh definitions
    if 'FlexCNN_for_Medical_Physics.classes.generators' in sys.modules:
        generators_module = importlib.reload(sys.modules['FlexCNN_for_Medical_Physics.classes.generators'])
        Gen_180 = generators_module.Generator_180
        Gen_256 = generators_module.Generator_256
        Gen_288 = generators_module.Generator_288
        Gen_320 = generators_module.Generator_320
    else:
        from FlexCNN_for_Medical_Physics.classes.generators import Generator_180 as Gen_180, Generator_256 as Gen_256, Generator_288 as Gen_288, Generator_320 as Gen_320

    checkpoint_path = os.path.join(paths['checkpoint_dirPath'], checkpoint_name)
    net_size = config['gen_sino_size']
    
    print(f"[DEBUG] Creating generator: gen_sino_size={net_size}")
    
    if net_size == 180:
        gen = Gen_180(config=config, gen_SI=True).to(device)
    elif net_size == 256:
        gen = Gen_256(config=config, gen_SI=True).to(device)
    elif net_size == 288:
        gen = Gen_288(config=config, gen_SI=True).to(device)
    elif net_size == 320:
        gen = Gen_320(config=config, gen_SI=True).to(device)
    else:
        raise ValueError(f"No Generator class available for gen_sino_size={net_size}. Supported sizes: 180, 256, 288, 320")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        gen.load_state_dict(checkpoint['gen_state_dict'])
    except RuntimeError as e:
        error_msg = str(e)
        if 'size mismatch' in error_msg or 'Missing key' in error_msg or 'Unexpected key' in error_msg:
            print(f"\n[ERROR] Configuration mismatch between checkpoint and provided config.")
            print(f"The config you provided does not match the config used to train '{checkpoint_name}'.")
            print(f"\nTo fix this:")
            print(f"1. Use the SAME config dictionary that was used to train this checkpoint")
            print(f"2. Or load a checkpoint trained with the current config")
            print(f"\nDetails:\n{error_msg}")
            raise
        else:
            raise
    gen.eval()
    with torch.no_grad():
        return gen(input_tensor).detach()

# Dual-network reconstruction (FROZEN_COFLOW, FROZEN_COUNTERFLOW)
def cnn_reconstruct_dual(
    attenuation_input,
    activity_input,
    config,
    paths,
    device,
    checkpoint_name
):
    """
    Dual-network reconstruction using frozen attenuation and trainable activity generators.
    attenuation_input: input for the attenuation network (atten_sino for coflow, atten_image for counterflow)
    activity_input: input for the activity network (always act_sino)
    """
    if config['network_type'] == 'FROZEN_COFLOW':
        flow_mode = 'coflow'
    elif config['network_type'] == 'FROZEN_COUNTERFLOW':
        flow_mode = 'counterflow'
    else:
        raise ValueError(f"Invalid network_type '{config['network_type']}' for dual-generator reconstruction.")

    # Instantiate both generators
    gen_atten, gen_act = instantiate_dual_generators(config, device, flow_mode)

    # Build checkpoint paths
    checkpoint_path_act = os.path.join(paths['checkpoint_dirPath'], checkpoint_name + '-act')
    checkpoint_path_atten = os.path.join(paths['checkpoint_dirPath'], checkpoint_name + '-atten')

    # Load checkpoints
    gen_act, gen_atten = load_dual_generator_checkpoints(
        gen_act,
        gen_atten,
        checkpoint_path_act,
        checkpoint_path_atten,
        load_state=True, # Loads both activity & attenuation states as long as checkpoints exist
        run_mode='test', # Puts activity network in eval mode
        device=device,
    )

    # Forward pass: get features from frozen attenuation network
    with torch.no_grad():
        atten_result = gen_atten(attenuation_input, return_features=True)
        frozen_enc_feats = atten_result['encoder']
        frozen_dec_feats = atten_result['decoder']

        # Activity network forward pass with injected features
        recon = gen_act(
            activity_input,
            frozen_encoder_features=frozen_enc_feats,
            frozen_decoder_features=frozen_dec_feats,
        ).detach()

    return recon

# Option B: User-selectable outputs to plot

def PlotPhantomRecons(indexes, checkpoint_name, network_type, 
                      config, paths, fig_size, device, settings, outputs_to_plot,
                      act_image_array_name=None, act_sino_array_name=None, atten_image_array_name=None,
                      atten_sino_array_name=None, recon1_array_name=None, recon2_array_name=None,
                      sino_resize_type='pool', sino_pad_type='zeros', image_pad_type='zeros',
                      sino_init_vert_cut=None, vert_pool_size=1, horiz_pool_size=1, bilinear_intermediate_size=161):
    """
    Load data, reconstruct images using a trained CNN, and visualize results.
    
    Supports single-network (ACT, ATTEN, CONCAT) and dual-network (FROZEN_COFLOW, FROZEN_COUNTERFLOW)
    architectures. Materializes config parameters, loads data from disk, selects appropriate network
    inputs based on architecture type, runs inference, and optionally visualizes selected outputs.
    
    Parameters
    ----------
    indexes : list of int
        Sample indices to load and reconstruct from the dataset.
    checkpoint_name : str
        Name of the checkpoint file (without extension) to load. For dual-network types,
        expects checkpoints named '{checkpoint_name}-act' and '{checkpoint_name}-atten'.
    network_type : str
        Network architecture type. Options:
        - 'ACT': Activity network (sinogram -> image)
        - 'ATTEN': Attenuation network (sino -> atten image)
        - 'CONCAT': Concatenated dual-input network
        - 'FROZEN_COFLOW': Frozen attenuation + trainable activity (coflow configuration)
        - 'FROZEN_COUNTERFLOW': Frozen attenuation + trainable activity (counterflow configuration)
        If None, uses config['network_type'].
    config : dict
        Hyperparameter dictionary containing network configuration (gen_sino_size, channels, 
        activation types, etc.). Supports string activation names which are converted to 
        PyTorch objects via materialize_config().
    paths : dict
        Dictionary with keys:
        - 'data_dirPath': Directory containing .npy data files
        - 'checkpoint_dirPath': Directory containing saved checkpoint files
    fig_size : int or tuple
        Figure size for visualization (passed to show_multiple_unmatched_tensors).
    device : str
        PyTorch device ('cpu' or 'cuda').
    settings : dict
        Data normalization and scaling factors (e.g., 'act_sino_scale', 'act_image_scale').
    outputs_to_plot : list of str or None
        List of output names to visualize. Options: 'act_image', 'act_sino', 'atten_image',
        'atten_sino', 'recon1', 'recon2', 'cnn_output'. If None, defaults to 
        ['act_image', 'cnn_output'].
    act_image_array_name : str, optional
        Filename of activity image array (.npy file).
    act_sino_array_name : str, optional
        Filename of activity sinogram array (.npy file).
    atten_image_array_name : str, optional
        Filename of attenuation image array (.npy file).
    atten_sino_array_name : str, optional
        Filename of attenuation sinogram array (.npy file).
    recon1_array_name : str, optional
        Filename of first reconstruction reference array (.npy file).
    recon2_array_name : str, optional
        Filename of second reconstruction reference array (.npy file).
    sino_resize_type : str, optional
        Sinogram resize method: 'pool' or 'bilinear'. Default: 'pool'.
    sino_pad_type : str, optional
        Sinogram padding type: 'zeros' or 'sinogram' (mirror/flip). Default: 'zeros'.
    image_pad_type : str, optional
        Image padding type: 'zeros' or 'none' (bilinear resize). Default: 'zeros'.
    sino_init_vert_cut : int, optional
        Symmetrically crop sinograms to this height before resizing. If None, no initial crop applied.
        Works with both 'pool' and 'bilinear' paths. Default: None.
    vert_pool_size : int, optional
        Vertical pooling factor for sinograms (1 = no pooling). Default: 1.
    horiz_pool_size : int, optional
        Horizontal pooling factor for sinograms (1 = no pooling). Default: 1.
    bilinear_intermediate_size : int, optional
        Intermediate size for bilinear resize before padding. Default: 161.
    
    Returns
    -------
    tensors : dict
        Dictionary of loaded and processed data tensors:
        - 'act_image_tensor': Shape (N, C, H, W) or None
        - 'act_sino_tensor': Shape (N, C, H, W) or None
        - 'atten_image_tensor': Shape (N, C, H, W) or None
        - 'atten_sino_tensor': Shape (N, C, H, W) or None
        - 'recon1_tensor': Shape (N, C, H, W) or None
        - 'recon2_tensor': Shape (N, C, H, W) or None
    cnn_output : torch.Tensor or None
        Network reconstruction output, shape (N, C, H, W) where N = len(indexes).
        None if no reconstruction was computed.
    
    Notes
    -----
    - Config parameters are automatically materialized (string activations converted to objects).
    - Generator classes are force-reloaded to handle Jupyter caching issues.
    - Config must match the checkpoint's original training config for successful loading.
    - For dual-network types, frozen attenuation features are passed to the activity network.
    - Large datasets should use device='cpu' if memory-constrained.
    
    Raises
    ------
    ValueError
        If network_type is unsupported or checkpoint config mismatches.
    RuntimeError
        If checkpoint state dict does not match the instantiated network architecture.
    
    Examples
    --------
    >>> tensors, recon = PlotPhantomRecons(
    ...     indexes=[0, 1, 2],
    ...     checkpoint_name='checkpoint-ACT-180-tuned',
    ...     network_type='ACT',
    ...     config=config_dict,
    ...     paths={'data_dirPath': '/path/to/data', 'checkpoint_dirPath': '/path/to/checkpoints'},
    ...     fig_size=4,
    ...     device='cpu',
    ...     settings=settings_dict,
    ...     outputs_to_plot=['act_image', 'act_sino', 'cnn_output'],
    ...     act_image_array_name='train-actMap.npy',
    ...     act_sino_array_name='train-highCountSino-320x257.npy'
    ... )
    """
    if network_type is None:
        network_type = config['network_type']
    # Materialize config to convert string activations to actual PyTorch objects
    config = materialize_config(config)
    tensors = BuildTensors(act_image_array_name, act_sino_array_name, atten_image_array_name, atten_sino_array_name,
                           recon1_array_name, recon2_array_name, config, paths, indexes, device, settings,
                           sino_resize_type=sino_resize_type, sino_pad_type=sino_pad_type, image_pad_type=image_pad_type,
                           sino_init_vert_cut=sino_init_vert_cut, vert_pool_size=vert_pool_size, horiz_pool_size=horiz_pool_size,
                           bilinear_intermediate_size=bilinear_intermediate_size)

    # Choose input for reconstruction based on network_type
    if network_type == 'ACT':
        input = tensors['act_sino_tensor']
    elif network_type == 'ATTEN':
        input = tensors['atten_sino_tensor']
    elif network_type == 'CONCAT':
        # Concatenate activity and attenuation sinograms along channel dim
        input = torch.cat([tensors['act_sino_tensor'], tensors['atten_sino_tensor']], dim=1)
    elif network_type == 'FROZEN_COFLOW':
        input = (tensors['atten_sino_tensor'], tensors['act_sino_tensor'])
    elif network_type == 'FROZEN_COUNTERFLOW':
        input = (tensors['atten_image_tensor'], tensors['act_sino_tensor'])
    else:
        raise ValueError(f"Unknown network_type: {network_type}")

    cnn_output = None
    if network_type == 'ACT' or network_type == 'ATTEN' or network_type == 'CONCAT':
        cnn_output = cnn_reconstruct_single(input, config, paths, device, checkpoint_name)
    elif network_type in ('FROZEN_COFLOW', 'FROZEN_COUNTERFLOW'):
        cnn_output = cnn_reconstruct_dual(input[0], input[1], config, paths, device, checkpoint_name)
    else:
        raise ValueError(f"Unknown network_type: {network_type}")

    # Build list of tensors to plot based on outputs_to_plot
    plot_map = {
        'act_image': [tensors['act_image_tensor']] if tensors['act_image_tensor'] is not None else [],
        'act_sino': [tensors['act_sino_tensor']] if tensors['act_sino_tensor'] is not None else [],
        'atten_image': [tensors['atten_image_tensor']] if tensors['atten_image_tensor'] is not None else [],
        'atten_sino': [tensors['atten_sino_tensor']] if tensors['atten_sino_tensor'] is not None else [],
        'recon1': [tensors['recon1_tensor']] if tensors['recon1_tensor'] is not None else [],
        'recon2': [tensors['recon2_tensor']] if tensors['recon2_tensor'] is not None else [],
        'cnn_output': [cnn_output] if cnn_output is not None else [],
    }
    tensors_to_plot = []
    if outputs_to_plot is None:
        outputs_to_plot = ['act_image', 'cnn_output']
    for key in outputs_to_plot:
        tensors_to_plot.extend(plot_map.get(key, []))

    show_multiple_unmatched_tensors(*tensors_to_plot, fig_size=fig_size)

    show_multiple_commonmap_tensors(*tensors_to_plot)

    return tensors, cnn_output