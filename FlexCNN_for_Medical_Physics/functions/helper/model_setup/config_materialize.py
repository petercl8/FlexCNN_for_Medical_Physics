"""
Convert string references in config dicts to actual PyTorch objects.

This allows tuning configs to use string names (which Optuna can serialize),
then materialize them to actual objects before training.
"""

from torch import nn
from FlexCNN_for_Medical_Physics.classes.losses import PatchwiseMomentLoss, VarWeightedMSE

# Constants for loss creation (from search_spaces.py)
PATCH_SIZE = 8
STRIDE = 4
MAX_MOMENT = 3
SCALE = 'mean'
COUNTS_PER_BQ = 60.0

# Custom losses that need special constructors
CUSTOM_LOSSES = {
    'VarWeightedMSE': lambda: VarWeightedMSE(k=COUNTS_PER_BQ),
    'PatchwiseMomentLoss': lambda: PatchwiseMomentLoss(
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        max_moment=MAX_MOMENT,
        scale=SCALE,
        weights=None
    ),
}


def materialize_config(config):
    """
    Convert string criterion/activation names in config to actual PyTorch objects.
    
    Automatically looks up strings in torch.nn first, then custom losses.
    Only strings in keys containing 'criterion' or 'activ' are converted.
    
    Args:
        config: Dict with potential string values for criteria and activations
    
    Returns:
        New dict with string values converted to instantiated objects
    """
    config = config.copy()  # Don't modify original
    
    for key in config:
        value = config[key]
        
        # Skip if already an object (not a string). Leaves existing objects intact.
        if not isinstance(value, str) or value is None or value == 'None':
            continue
        
        # Only convert criterion and activation keys
        if 'criterion' not in key.lower() and 'activ' not in key.lower():
            continue
        
        # Try custom losses first
        if value in CUSTOM_LOSSES:
            config[key] = CUSTOM_LOSSES[value]()
        # Try torch.nn
        elif hasattr(nn, value):
            cls = getattr(nn, value)
            config[key] = cls()
        # If not found, leave as string (might be a placeholder or flag)
        else:
            print(f"[WARNING] Could not find criterion/activation '{value}' - leaving as string")
    
    return config
