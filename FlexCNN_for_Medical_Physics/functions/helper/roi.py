import numpy as np
import torch


def _to_numpy(x):
    """Convert tensor or numpy array to numpy, handling device placement."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ROI_NEMA_hot(ground_truth_tensor, reconstruction_tensor, background_mask, hot_mask):
    # Convert all inputs to numpy
    background_mask = _to_numpy(background_mask)
    ground_truth_tensor = _to_numpy(ground_truth_tensor)
    reconstruction_tensor = _to_numpy(reconstruction_tensor)
    hot_mask = _to_numpy(hot_mask)
   
    background_truth = np.sum(np.multiply(background_mask, ground_truth_tensor))
    background_reconstruction = np.sum(np.multiply(background_mask, reconstruction_tensor))

    hot_truth = np.sum(np.multiply(hot_mask, ground_truth_tensor))
    hot_reconstruction = np.sum(np.multiply(hot_mask, reconstruction_tensor))

    x = 100*(hot_reconstruction/background_reconstruction-1)/(hot_truth/background_truth-1)

    return x


def ROI_NEMA_cold(reconstruction_tensor, background_mask, cold_mask):
    # Convert all inputs to numpy
    background_mask = _to_numpy(background_mask)
    reconstruction_tensor = _to_numpy(reconstruction_tensor)
    cold_mask = _to_numpy(cold_mask)

    background_reconstruction = np.sum(np.multiply(background_mask, reconstruction_tensor))
    cold_reconstruction = np.sum(np.multiply(cold_mask, reconstruction_tensor))

    x = 100*(1-cold_reconstruction/background_reconstruction)

    return x