import os
import torch
import numpy as np

from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataLoader
from FlexCNN_for_Medical_Physics.classes.generators import Generator_180, Generator_288, Generator_320
from FlexCNN_for_Medical_Physics.functions.helper.display_images import show_multiple_unmatched_tensors


def BuildImageSinoTensors(image_array_names, sino_array_name, config, paths_dict, indexes, device, settings):
    # --- Normalize input so we always have a list of image array names ---
    if isinstance(image_array_names, str):
        image_array_names = [image_array_names]

    # --- Load the sinogram array (only one) ---
    sino_array_path = os.path.join(paths_dict['data_dirPath'], sino_array_name)
    sino_array = np.load(sino_array_path, mmap_mode='r')

    # --- Load each image array into memory-mapped numpy objects ---
    image_arrays = []
    for name in image_array_names:
        path = os.path.join(paths_dict['data_dirPath'], name)
        image_arrays.append(np.load(path, mmap_mode='r'))

    # --- Prepare empty lists to collect image tensors ---
    image_tensors = []
    sino_tensor = None
    first_sino = True


    # --- Loop over each image array separately ---
    for array_num, image_array in enumerate(image_arrays):
        # Build tensors for this image array
        i = 0
        for idx in indexes:
            sino_ground_scaled, image_ground_scaled, _, _ = NpArrayDataLoader(
                image_array, sino_array, config, settings, augment=(None, False), index=idx, device=device
            )

            if first_sino:
                # create the sino tensor only once
                sino_tensor = torch.zeros(
                    len(indexes),
                    sino_ground_scaled.shape[0],
                    sino_ground_scaled.shape[1],
                    sino_ground_scaled.shape[2],
                    dtype=torch.float32,
                    device=device
                )
                first_sino = False

            # create a fresh image tensor for *this* image array
            if i == 0:
                image_tensor = torch.zeros(
                    len(indexes),
                    image_ground_scaled.shape[0],
                    image_ground_scaled.shape[1],
                    image_ground_scaled.shape[2],
                    dtype=torch.float32,
                    device=device
                )

            image_tensor[i, :] = image_ground_scaled

            if array_num == 0:
                sino_tensor[i, :] = sino_ground_scaled

            i += 1

        image_tensors.append(image_tensor)
        first = False  # only create sino_tensor on first pass

    return image_tensors, sino_tensor

def CNN_reconstruct(sino_tensor, config, checkpoint_name, paths, device, sino_size=288):
    checkpoint_path = os.path.join(paths['checkpoint_dirPath'], checkpoint_name)
    if sino_size == 180:
        gen = Generator_180(config=config, gen_SI=True).to(device)
    elif sino_size == 288:
        gen = Generator_288(config=config, gen_SI=True).to(device)
    elif sino_size == 320:
        gen = Generator_320(config=config, gen_SI=True).to(device)
    else:
        raise ValueError(f"No Generator class available for sino_size={sino_size}. Supported sizes: 180, 288, 320")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen.eval()
    with torch.no_grad():
        return gen(sino_tensor).detach()


def PlotPhantomRecons(image_array_names, sino_array_name, config, paths_dict, indexes, checkpointName, fig_size, device, settings):
    image_tensors, sino_tensor = BuildImageSinoTensors(image_array_names, sino_array_name, config, paths_dict, indexes, device, settings)
    CNN_output = CNN_reconstruct(sino_tensor, config, checkpointName, paths_dict, device)
    show_multiple_unmatched_tensors(*image_tensors, CNN_output, fig_size=fig_size)
    return image_tensors, sino_tensor

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