import os
import torch
from FlexCNN_for_Medical_Physics.functions.main_run_functions.train_utils import create_generator, init_checkpoint_state
from FlexCNN_for_Medical_Physics.functions.helper.weights_init import weights_init_he

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
    checkpoint_path,
    load_state,
    run_mode,
    device,
):
    """
    Loads or initializes checkpoints/weights for dual-generator setup (activity and attenuation networks).
    Args:
        gen_act: Activity generator model (trainable)
        gen_atten: Attenuation generator model (frozen)
        checkpoint_path: Base path for checkpoints
        load_state: Whether to load activity generator state
        run_mode: Current run mode (train/test/visualize/tune)
        device: Torch device
    Returns:
        gen_act, gen_atten
    """
    from FlexCNN_for_Medical_Physics.functions.helper.weights_init import weights_init_he
    import torch
    import os

    act_ckpt = checkpoint_path + '-act' if checkpoint_path is not None else None
    atten_ckpt = checkpoint_path + '-atten' if checkpoint_path is not None else None

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