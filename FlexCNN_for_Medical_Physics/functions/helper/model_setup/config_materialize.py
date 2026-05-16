"""
Convert string references in config dicts to actual PyTorch objects.

This allows tuning configs to use string names (which Optuna can serialize),
then materialize them to actual objects before training.
"""

from torch import nn
from FlexCNN_for_Medical_Physics.custom_criteria import PatchwiseMomentLoss, VarWeightedMSE

# Custom losses that need special constructors
CUSTOM_LOSSES = {
    # Both losses use defaults from losses/defaults.py (single source of truth)
    'VarWeightedMSE': lambda: VarWeightedMSE(),
    'PatchwiseMomentLoss': lambda: PatchwiseMomentLoss(),
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

    # ------------------------------------------------------------------
    # Normalization pass (backwards compatibility)
    # - Convert Python None -> 'none' sentinel so old saved configs
    #   that used None remain compatible with the newer string-based
    #   sentinel policy.
    # - Convert string booleans 'true'/'false' (from persisted search
    #   spaces) to Python bools so downstream code that expects bools
    #   continues to work.
    # - Leave existing Python True/False values alone.
    # ------------------------------------------------------------------
    # Define explicit whitelist of key suffixes that use the 'none' sentinel.
    # This prevents accidental conversion of unrelated None defaults (e.g. numeric
    # fallbacks) into the string 'none'. Use suffixes so SI_/IS_ prefixes work.
    sentinel_suffixes = ('_final_activ', '_final_activation', '_criterion', '_skip_mode')

    def key_uses_sentinel(k: str) -> bool:
        kl = k.lower()
        return any(kl.endswith(suf) for suf in sentinel_suffixes)

    for key, value in list(config.items()):
        # Only normalize Python None -> 'none' for whitelisted sentinel keys.
        if value is None:
            if key_uses_sentinel(key):
                config[key] = 'none'
            # otherwise leave None as-is (used as an actual None sentinel by runtime)
            continue

        # Normalize textual booleans to actual bool type (case-insensitive)
        if isinstance(value, str):
            low = value.lower()
            if low == 'true':
                config[key] = True
                continue
            if low == 'false':
                config[key] = False
                continue
            # Normalize textual 'none' only for whitelisted sentinel keys
            if low == 'none' and key_uses_sentinel(key):
                config[key] = 'none'
                continue

    # ------------------------------------------------------------------
    # Materialize only criterion/activation keys: convert string names
    # like 'Tanh' or 'LeakyReLU' into instantiated `torch.nn` modules,
    # or custom losses into their constructed objects.
    # Note: we intentionally do NOT materialize general booleans or
    # non-relevant keys here — the normalization pass above handles
    # boolean strings and None sentinels.
    # ------------------------------------------------------------------
    for key in list(config.keys()):
        value = config[key]

        # Only convert criterion and activation keys
        if 'criterion' not in key.lower() and 'activ' not in key.lower():
            continue

        # If value is not a string (already an object or a boolean), skip
        if not isinstance(value, str):
            continue

        # Skip the 'none' sentinel (explicit no-op)
        if value.lower() == 'none':
            continue

        # Try custom losses first
        if value in CUSTOM_LOSSES:
            config[key] = CUSTOM_LOSSES[value]()
        # Try torch.nn by attribute name
        elif hasattr(nn, value):
            cls = getattr(nn, value)
            config[key] = cls()
        # If not found, leave as string (might be a placeholder or flag)
        else:
            print(f"[WARNING] Could not find criterion/activation '{value}' - leaving as string")

    return config
