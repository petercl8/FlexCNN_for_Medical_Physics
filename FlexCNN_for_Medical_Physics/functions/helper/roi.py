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



def ROI_simple_phantom(ground_truth_tensor, reconstruction_tensor, hot_mask):
    """
    Symmetric contrast recovery with separate error metrics:
    - Hot underestimation in hot region
    - Cold region overestimation (hot leaking into cold)
    - Combined CR
    Handles residual activity in the cold region due to voxelization.

    Parameters:
        ground_truth_tensor: np.array or torch tensor with true activity
        reconstruction_tensor: np.array or torch tensor with reconstructed activity
        hot_mask: binary mask of the hot region (1 inside hot, 0 elsewhere)

    Returns:
        dict with keys:
            'CR_symmetric': combined contrast recovery (%)
            'hot_underestimation': hot underestimation (%)
            'cold_overestimation': cold region overestimation (%)
    """
    # Convert to numpy if needed
    hot_mask = _to_numpy(hot_mask)
    ground_truth_tensor = _to_numpy(ground_truth_tensor)
    reconstruction_tensor = _to_numpy(reconstruction_tensor)

    # Cold mask is complement of hot
    cold_mask = 1 - hot_mask

    # Voxel counts
    V_hot = np.sum(hot_mask)
    V_cold = np.sum(cold_mask)

    # Mean values
    R_hot_mean = np.sum(reconstruction_tensor * hot_mask) / V_hot
    R_cold_mean = np.sum(reconstruction_tensor * cold_mask) / V_cold

    T_hot_mean = np.sum(ground_truth_tensor * hot_mask) / V_hot
    T_cold_mean = np.sum(ground_truth_tensor * cold_mask) / V_cold

    # Symmetric contrast recovery
    CR_symmetric = 100 * (R_hot_mean - R_cold_mean) / (T_hot_mean - T_cold_mean + 1e-12)

    # Separate penalties
    hot_underestimation = 100 * (T_hot_mean - R_hot_mean) / (T_hot_mean + 1e-12)
    cold_overestimation = 100 * (R_cold_mean - T_cold_mean) / (T_hot_mean - T_cold_mean + 1e-12)

    return {
        'CR_symmetric': CR_symmetric,
        'hot_underestimation': hot_underestimation,
        'cold_overestimation': cold_overestimation
    }