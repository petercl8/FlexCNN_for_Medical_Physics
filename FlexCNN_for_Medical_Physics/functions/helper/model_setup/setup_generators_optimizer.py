import os
import torch
from FlexCNN_for_Medical_Physics.functions.helper.model_setup.weights_init import weights_init_he
from FlexCNN_for_Medical_Physics.classes.generators import Generator_180, Generator_256, Generator_288, Generator_320

def create_generator(config: dict, device: str, **kwargs):
    """
    Instantiate the appropriate Generator class based on max(gen_image_size, gen_sino_size).
    
    The generator is sized to the larger dimension to support flexible input/output handling.
    For example, if sizes are 180 and 256, Generator_256 is used regardless of train_SI direction.
    
    Args:
        config: Configuration dictionary containing 'gen_image_size', 'gen_sino_size', 'train_SI'.
        device: Device to place generator on (e.g., 'cuda:0' or 'cpu').
        **kwargs: Additional keyword arguments passed to Generator constructor (e.g., gen_skip_handling, frozen_enc_channels).
    
    Returns:
        Generator instance (Generator_180, Generator_256, Generator_288, or Generator_320) on specified device.
    
    Raises:
        ValueError if max size is not 180, 256, 288, or 320.
    """
    train_SI = config['train_SI']
    gen_image_size = config['gen_image_size']
    gen_sino_size = config['gen_sino_size']

    # Select generator based on maximum dimension (input or output, regardless of direction)
    # This ensures the generator can handle both sizes in either direction with optional padding
    max_size = max(gen_image_size, gen_sino_size)
    
    # Select appropriate Generator class based on max size
    if max_size == 180:
        GeneratorClass = Generator_180
    elif max_size == 256:
        GeneratorClass = Generator_256
    elif max_size == 288:
        GeneratorClass = Generator_288
    elif max_size == 320:
        GeneratorClass = Generator_320
    else:
        raise ValueError(f"No Generator class available for max_size={max_size}. Supported sizes: 180, 256, 288, 320")
    
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


def create_lr_scheduler(optimizer, settings: dict, total_epochs: int, resumed_epochs: int = 0, advance_on_resume: bool = True):
    """
    Create an epoch-based learning rate scheduler for training mode.

    Args:
        optimizer: Optimizer instance.
        settings: Runtime settings dictionary.
        total_epochs: Total planned training epochs (T_max for cosine schedule).
        resumed_epochs: Number of completed epochs before current run.
        advance_on_resume: If True, manually advance scheduler by resumed_epochs.
            Use False when loading scheduler state from checkpoint.

    Returns:
        Scheduler instance or None when scheduling is disabled.
    """
    if settings.get('run_mode') != 'train':
        return None

    schedule_type = settings.get('train_lr_schedule_type')
    if schedule_type == 'none':
        return None

    if schedule_type != 'cosine':
        raise ValueError(f"Unknown train_lr_schedule_type '{schedule_type}'.")

    if total_epochs <= 0:
        raise ValueError(f"total_epochs must be positive, got {total_epochs}.")

    min_factor = settings.get('train_lr_min_factor')
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    eta_min = min(base_lrs) * min_factor

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=eta_min,
    )

    # Backward-compatible resume path for checkpoints without scheduler state.
    # Only advance when explicitly requested by caller.
    if advance_on_resume:
        for _ in range(max(0, resumed_epochs)):
            scheduler.step()

    return scheduler

def _extract_frozen_config(config):
    """
    Strip FROZEN_ prefix from all keys that have it to prepare config for frozen generator.
    
    Args:
        config: Full merged configuration dictionary with FROZEN_* keys for frozen network
                and unprefixed keys for trainable network.
    
    Returns:
        New dictionary with FROZEN_ prefix stripped from all matching keys.
    
    Example:
        {'FROZEN_SI_gen_neck': 'narrow', 'SI_gen_neck': 'medium', 'FROZEN_gen_sino_size': 288}
        -> {'SI_gen_neck': 'narrow', 'gen_sino_size': 288}
    """
    frozen_config = {}
    for k, v in config.items():
        if k.startswith('FROZEN_'):
            frozen_config[k.replace('FROZEN_', '', 1)] = v
    return frozen_config

def instantiate_dual_generators(config, device, flow_mode):
    """
    Instantiate frozen attenuation and trainable activity generators for dual-network setup.

    Args:
        config (dict): Model configuration dictionary with FROZEN_* prefixed keys for frozen network.
        device (torch.device): Device to place models on.
        flow_mode (str): 'coflow' or 'counterflow'.

    Returns:
        (gen_frozen, gen_act): Tuple of instantiated generators.

    Example:
        gen_frozen, gen_act = instantiate_dual_generators(config, device, flow_mode)
    """
    # Extract frozen network config by stripping FROZEN_ prefix
    frozen_config = _extract_frozen_config(config)
    frozen_variant = str(config.get('frozen_variant')).upper()
    if frozen_variant not in ('ATTEN', 'RECON_SINO'):
        raise ValueError(f"Invalid frozen_variant='{config.get('frozen_variant')}'. Expected ATTEN or RECON_SINO.")
    expected_frozen_type = 'ATTEN' if frozen_variant == 'ATTEN' else 'RECON_SINO'
    loaded_frozen_type = str(frozen_config.get('network_type')).upper()
    if loaded_frozen_type != expected_frozen_type:
        raise ValueError(
            f"Frozen config mismatch: loaded frozen network_type='{loaded_frozen_type}' "
            f"but frozen_variant='{frozen_variant}'."
        )
    frozen_config['train_SI'] = True if flow_mode == 'coflow' else False  # Guarantee network direction regardless of value set in notebook
    gen_frozen = create_generator(frozen_config, device, gen_skip_handling='1x1Conv', gen_flow_mode='coflow', frozen_enc_channels=None, frozen_dec_channels=None)  # Always 'coflow' to match standalone training

    # Extract encoder/decoder channel tuples for injection into activity network
    enc_inject_ch = gen_frozen.enc_stage_channels
    dec_inject_ch = gen_frozen.dec_stage_channels

    # Create trainable activity generator with unprefixed SI_* keys from original config
    act_config = dict(config)
    act_config['train_SI'] = True
    
    gen_act = create_generator(act_config, device, gen_skip_handling='1x1Conv', gen_flow_mode=flow_mode, frozen_enc_channels=enc_inject_ch, frozen_dec_channels=dec_inject_ch)
    return gen_frozen, gen_act


def load_dual_generator_checkpoints(
    gen_act,
    gen_frozen,
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
        gen_frozen: Frozen generator model
        act_ckpt: Checkpoint path for activity generator
        atten_ckpt: Checkpoint path for attenuation generator
        load_state: Whether to load activity generator state
        run_mode: Current run mode (train/test/visualize/tune)
        device: Torch device
    Returns:
        gen_act, gen_frozen
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

    # Always load frozen checkpoint if available
    if atten_ckpt is not None and os.path.exists(atten_ckpt):
        atten_checkpoint = torch.load(atten_ckpt, map_location=device)
        gen_frozen.load_state_dict(atten_checkpoint['gen_state_dict'])
    else:
        raise FileNotFoundError(f"Frozen checkpoint not found at '{atten_ckpt}' for {run_mode}.")
    
    gen_frozen.eval()

    # Set eval mode for test/visualize/plotting
    if run_mode in ('test', 'visualize'):
        gen_act.eval()

    return gen_act, gen_frozen
