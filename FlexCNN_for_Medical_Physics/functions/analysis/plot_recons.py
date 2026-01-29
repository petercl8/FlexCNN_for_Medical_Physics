import os
import torch
import numpy as np

from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataLoader
from FlexCNN_for_Medical_Physics.classes.generators import Generator_180, Generator_288, Generator_320
from FlexCNN_for_Medical_Physics.functions.helper.setup_generators_optimizer import instantiate_dual_generators, load_dual_generator_checkpoints
from FlexCNN_for_Medical_Physics.functions.helper.display_images import show_multiple_unmatched_tensors



# Updated: BuildImageSinoTensors returns lists of all images and sinograms (activity, attenuation), and reconstructions if present

def BuildTensors(act_image_array_name, act_sino_array_name, atten_image_array_name, atten_sino_array_name, recon1_array_name, recon2_array_name, config, paths_dict, indexes, device, settings):
    '''
    Return a dictionary of tensors for activity images/sinograms, attenuation images/sinograms, and reconstructions.
    Each tensor has shape (N, C, H, W), where N is the number of indexes provided (len(indexes)).
    '''
    def load_array(name):
        if name is None:
            return None
        return np.load(os.path.join(paths_dict['data_dirPath'], name), mmap_mode='r')

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
            config, settings, augment=(None, False), index=idx, device=device
        )
        act_sino_scaled, act_image_scaled = act_data
        atten_sino_scaled, atten_image_scaled = atten_data
        act_recon1, act_recon2 = recon_data

        act_image_tensors.append(act_image_scaled)
        act_sino_tensors.append(act_sino_scaled)
        atten_image_tensors.append(atten_image_scaled)
        atten_sino_tensors.append(atten_sino_scaled)
        recon1_tensors.append(act_recon1)
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

    checkpoint_path = os.path.join(paths['checkpoint_dirPath'], checkpoint_name)
    net_size = config['gen_sino_size']
    if net_size == 180:
        gen = Generator_180(config=config, gen_SI=True).to(device)
    elif net_size == 288:
        gen = Generator_288(config=config, gen_SI=True).to(device)
    elif net_size == 320:
        gen = Generator_320(config=config, gen_SI=True).to(device)
    else:
        raise ValueError(f"No Generator class available for gen_sino_size={net_size}. Supported sizes: 180, 288, 320")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint['gen_state_dict'])
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

def PlotPhantomRecons(act_image_array_name, act_sino_array_name, atten_image_array_name, atten_sino_array_name, recon1_array_name, recon2_array_name,
                      indexes, checkpoint_name, network_type, 
                      config, paths_dict, fig_size, device, settings, outputs_to_plot=None):
    """
    outputs_to_plot: list of strings, e.g. ['act_image', 'act_sino', 'atten_image', 'atten_sino', 'recon1', 'recon2', 'cnn_output']
    network_type: if None, will use config['network_type']
    """
    if network_type is None:
        network_type = config['network_type']

    tensors = BuildTensors(act_image_array_name, act_sino_array_name, atten_image_array_name, atten_sino_array_name,
                            recon1_array_name, recon2_array_name, config, paths_dict, indexes, device, settings)

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
        cnn_output = cnn_reconstruct_single(input, config, paths_dict, device, checkpoint_name)
    elif network_type in ('FROZEN_COFLOW', 'FROZEN_COUNTERFLOW'):
        cnn_output = cnn_reconstruct_dual(input[0], input[1], config, paths_dict, device, checkpoint_name)
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
    return tensors, cnn_output