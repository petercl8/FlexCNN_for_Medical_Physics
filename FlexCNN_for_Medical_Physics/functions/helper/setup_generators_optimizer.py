import os
import torch
from FlexCNN_for_Medical_Physics.functions.helper.weights_init import weights_init_he

def create_generator(config: dict, device: str, **kwargs):
    """
    Instantiate the appropriate Generator class based on config input size.
    
    Args:
        config: Configuration dictionary containing 'gen_image_size', 'gen_sino_size', 'train_SI'.
        device: Device to place generator on (e.g., 'cuda:0' or 'cpu').
        **kwargs: Additional keyword arguments passed to Generator constructor (e.g., gen_skip_handling, enc_inject_channels).
    
    Returns:
        Generator instance (Generator_180, Generator_288, or Generator_320) on specified device.
    
    Raises:
        ValueError if input_size is not 180, 288, or 320.
    """
    train_SI = config['train_SI']
    gen_image_size = config['gen_image_size']
    gen_sino_size = config['gen_sino_size']
    
    # Determine input size based on train_SI direction
    input_size = gen_sino_size if train_SI else gen_image_size
    
    # Select appropriate Generator class
    if input_size == 180:
        GeneratorClass = Generator_180
    elif input_size == 288:
        GeneratorClass = Generator_288
    elif input_size == 320:
        GeneratorClass = Generator_320
    else:
        raise ValueError(f"No Generator class available for input_size={input_size}. Supported sizes: 180, 288, 320")
    
    # Instantiate and move to device
    gen = GeneratorClass(config=config, gen_SI=train_SI, **kwargs).to(device)
    return gen


def create_optimizer(model, config: dict) -> torch.optim.Adam:
    """
    Create optimizer for generator with optional separate learning rate group for scale parameters.
    
    Args:
        model: Generator model instance.
        config: Configuration dictionary containing 'gen_b1', 'gen_b2', 'gen_lr', 
                'SI_output_scale_lr_mult' or 'IS_output_scale_lr_mult', and 'train_SI'.
    
    Returns:
        torch.optim.Adam optimizer, potentially with parameter groups for learnable scale.
    """
    train_SI = config['train_SI']
    betas = (config['gen_b1'], config['gen_b2'])
    base_lr = config['gen_lr']
    
    # Determine scale LR multiplier based on direction
    if train_SI:
        scale_lr_mult = config['SI_output_scale_lr_mult']
    else:
        scale_lr_mult = config['IS_output_scale_lr_mult']
    
    # If model has learnable output scale, create separate param group for it
    if getattr(model, 'output_scale_learnable', False):
        scale_param = [model.log_output_scale]
        main_params = [p for n, p in model.named_parameters() if n != 'log_output_scale']
        optimizer = torch.optim.Adam(
            [
                {'params': main_params, 'lr': base_lr, 'betas': betas},
                {'params': scale_param, 'lr': scale_lr_mult * base_lr, 'betas': betas, 'weight_decay': 0.0},
            ]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=betas)
    
    return optimizer

def instantiate_dual_generators(config, device, flow_mode):
    """
    Instantiate frozen attenuation and trainable activity generators for dual-network setup.

    Args:
        config (dict): Model configuration dictionary.
        device (torch.device): Device to place models on.
        flow_mode (str): 'coflow' or 'counterflow'.

    Returns:
        (gen_atten, gen_act): Tuple of instantiated generators.

    Example:
        gen_atten, gen_act = instantiate_dual_generators(config, device, flow_mode)
    """
    # Create frozen attenuation generator (no injection, only feature extraction)
    atten_config = dict(config)
    atten_config['train_SI'] = True if flow_mode == 'coflow' else False  # Guarantee network direction regardless of value set in notebook
    gen_atten = create_generator(atten_config, device, gen_skip_handling='1x1Conv', enc_inject_channels=None, dec_inject_channels=None) # Default flow moade is 'coflow', but it doesn't matter since no features are injected

    # Extract encoder/decoder channel tuples for injection into activity network
    enc_inject_ch = gen_atten.enc_stage_channels
    dec_inject_ch = gen_atten.dec_stage_channels

    # Create trainable activity generator with injection parameters
    act_config = dict(config)
    act_config['train_SI'] = True
    
    gen_act = create_generator(act_config, device, gen_skip_handling='1x1Conv', gen_flow_mode=flow_mode, enc_inject_channels=enc_inject_ch, dec_inject_channels=dec_inject_ch)
    return gen_atten, gen_act


def load_dual_generator_checkpoints(
    gen_act,
    gen_atten,
    act_ckpt,
    atten_ckpt,
    load_state,
    run_mode,
    device,
):
    """
    Loads or initializes checkpoints/weights for dual-generator setup (activity and attenuation networks).
    Args:
        gen_act: Activity generator model (trainable)
        gen_atten: Attenuation generator model (frozen)
        act_ckpt: Checkpoint path for activity generator
        atten_ckpt: Checkpoint path for attenuation generator
        load_state: Whether to load activity generator state
        run_mode: Current run mode (train/test/visualize/tune)
        device: Torch device
    Returns:
        gen_act, gen_atten
    """
    # Activity generator: load or randomize weights
    if act_ckpt is not None and load_state and os.path.exists(act_ckpt):
        act_checkpoint = torch.load(act_ckpt, map_location=device)
        if 'gen_state_dict' in act_checkpoint:
            gen_act.load_state_dict(act_checkpoint['gen_state_dict'])
        else:
            gen_act = gen_act.apply(weights_init_he)
    else:
        gen_act = gen_act.apply(weights_init_he)

    # Always load attenuation checkpoint if available
    if atten_ckpt is not None and os.path.exists(atten_ckpt):
        atten_checkpoint = torch.load(atten_ckpt, map_location=device)
        gen_atten.load_state_dict(atten_checkpoint['gen_state_dict'])
    else:
        raise FileNotFoundError(f"Attenuation checkpoint not found at '{atten_ckpt}' for {run_mode}.")
    
    gen_atten.eval()

    # Set eval mode for test/visualize/plotting
    if run_mode in ('test', 'visualize'):
        gen_act.eval()

    return gen_act, gen_atten
