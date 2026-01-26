import os
import torch
import numpy as np

from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataLoader
from FlexCNN_for_Medical_Physics.classes.generators import Generator_180, Generator_288, Generator_320
from FlexCNN_for_Medical_Physics.functions.helper.display_images import show_multiple_unmatched_tensors



# Updated: BuildImageSinoTensors returns lists of all images and sinograms (activity, attenuation), and reconstructions if present

def BuildActivityTensors(act_image_array_name, act_sino_array_name, atten_image_array_name, atten_sino_array_name, recon1_array_name, recon2_array_name, config, paths_dict, indexes, device, settings):
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
        'act_image_tensors': torch.stack(act_image_tensors) if act_image_tensors else None,
        'act_sino_tensors': torch.stack(act_sino_tensors) if act_sino_tensors else None,
        'atten_image_tensors': torch.stack(atten_image_tensors) if atten_image_tensors else None,
        'atten_sino_tensors': torch.stack(atten_sino_tensors) if atten_sino_tensors else None,
        'recon1_tensors': torch.stack(recon1_tensors) if recon1_tensors else None,
        'recon2_tensors': torch.stack(recon2_tensors) if recon2_tensors else None,
    }


# Single-network reconstruction (SUP, ATTEN, CONCAT)
def cnn_reconstruct_single(input_tensor, config, checkpoint_name, paths, device, gen_type='SUP'):
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
def cnn_reconstruct_dual(input_tensor, config, paths, device, checkpoint_name_act, checkpoint_name_atten, feature_inject_to_encoder=True, feature_inject_to_decoder=True):
    # Load attenuation (frozen) network
    net_size = config['gen_sino_size']
    if net_size == 180:
        GenClass = Generator_180
    elif net_size == 288:
        GenClass = Generator_288
    elif net_size == 320:
        GenClass = Generator_320
    else:
        raise ValueError(f"No Generator class available for gen_sino_size={net_size}. Supported sizes: 180, 288, 320")
    # Instantiate generators with configs
    gen_act = GenClass(**gen_config_act).to(device)
    gen_atten = GenClass(**gen_config_atten).to(device)

    # Load checkpoints with device mapping
    checkpoint_act = torch.load(checkpoint_path_act, map_location=device)
    checkpoint_atten = torch.load(checkpoint_path_atten, map_location=device)
    gen_act.load_state_dict(checkpoint_act['gen_state_dict'])
    gen_atten.load_state_dict(checkpoint_atten['gen_state_dict'])
    gen_act.eval()
    gen_atten.eval()

    # Extract injection channels if present in config
    inj_channels = injection_channels
    if 'injection_channels' in gen_config_act:
        inj_channels = gen_config_act['injection_channels']

    # Forward pass
    with torch.no_grad():
        # Attenuation pass (frozen backbone)
        atten_features = gen_atten.forward_features(atten_tensor)
        # Activity pass (inject features)
        recon = gen_act.forward_with_injection(act_tensor, atten_features, inj_channels)
    return recon



# Option B: User-selectable outputs to plot

def PlotPhantomRecons(act_image_array_name, act_sino_array_name, atten_image_array_name, atten_sino_array_name, recon1_array_name, recon2_array_name,
                      config, paths_dict, indexes, checkpointName, fig_size, device, settings, network_type=None, outputs_to_plot=None,
                      checkpointName_atten=None, feature_inject_to_encoder=True, feature_inject_to_decoder=True):
    """
    outputs_to_plot: list of strings, e.g. ['act_image', 'act_sino', 'atten_image', 'atten_sino', 'recon1', 'recon2', 'cnn_output']
    network_type: if None, will use config['network_type']
    """
    if network_type is None:
        network_type = config['network_type']
    tensors = BuildActivityTensors(act_image_array_name, act_sino_array_name, atten_image_array_name, atten_sino_array_name, recon1_array_name, recon2_array_name, config, paths_dict, indexes, device, settings)

    # Choose input for reconstruction based on network_type
    if network_type in ('SUP', 'ACT'):
        input_tensor = tensors['act_sino_tensors'] if tensors['act_sino_tensors'] is not None else None
    elif network_type == 'ATTEN':
        input_tensor = tensors['atten_sino_tensors'] if tensors['atten_sino_tensors'] is not None else None
    elif network_type == 'CONCAT':
        # Concatenate activity and attenuation sinograms along channel dim
        if tensors['act_sino_tensors'] is not None and tensors['atten_sino_tensors'] is not None:
            input_tensor = torch.cat([tensors['act_sino_tensors'], tensors['atten_sino_tensors']], dim=1)
        else:
            input_tensor = None
    else:
        input_tensor = tensors['act_sino_tensors'] if tensors['act_sino_tensors'] is not None else None

    cnn_output = None
    if network_type in ('SUP', 'ACT', 'ATTEN', 'CONCAT'):
        cnn_output = cnn_reconstruct_single(input_tensor, config, checkpointName, paths_dict, device)
    elif network_type in ('FROZEN_COFLOW', 'FROZEN_COUNTERFLOW'):
        cnn_output = cnn_reconstruct_dual(input_tensor, config, paths_dict, device, checkpointName, checkpointName_atten, feature_inject_to_encoder, feature_inject_to_decoder)
    else:
        raise ValueError(f"Unknown network_type: {network_type}")

    # Build list of tensors to plot based on outputs_to_plot
    plot_map = {
        'act_image': [tensors['act_image_tensors']] if tensors['act_image_tensors'] is not None else [],
        'act_sino': [tensors['act_sino_tensors']] if tensors['act_sino_tensors'] is not None else [],
        'atten_image': [tensors['atten_image_tensors']] if tensors['atten_image_tensors'] is not None else [],
        'atten_sino': [tensors['atten_sino_tensors']] if tensors['atten_sino_tensors'] is not None else [],
        'recon1': [tensors['recon1_tensors']] if tensors['recon1_tensors'] is not None else [],
        'recon2': [tensors['recon2_tensors']] if tensors['recon2_tensors'] is not None else [],
        'cnn_output': [cnn_output] if cnn_output is not None else [],
    }
    tensors_to_plot = []
    if outputs_to_plot is None:
        outputs_to_plot = ['act_image', 'cnn_output']
    for key in outputs_to_plot:
        tensors_to_plot.extend(plot_map.get(key, []))

    show_multiple_unmatched_tensors(*tensors_to_plot, fig_size=fig_size)
    return tensors, cnn_output

'''
OLDER VERSION BELOW - TO BE DEPRECATED

## CNN Outputs ##
def CNN_reconstruct(sino_tensor, config, checkpoint_dirPath, checkpoint_fileName):

    #Construct CNN reconstructions of images of a sinogram tensor.
    #Config must contain: sino_size, sino_channels, image_channels.

    gen = Generator(config=config, gen_SI=True).to(device)
    checkpoint_path = os.path.join(checkpoint_dirPath, checkpoint_fileName)
    checkpoint = torch.load(checkpoint_path)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen.eval()
    return gen(sino_tensor).detach()
'''