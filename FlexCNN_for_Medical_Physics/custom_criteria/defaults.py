"""
Default parameter values for patchwise moment computation and loss functions.

This module serves as the single source of truth for all default values used in
the losses module. Change defaults here to affect all loss and metric functions.
"""

# ============================
# Patchwise Moment Parameters
# ============================

# Patch extraction parameters
PATCH_SIZE = 8
"""Default patch size (side length of square patches in pixels)."""

STRIDE = 4
"""Default stride for patch extraction (smaller = more overlap)."""

# Moment computation parameters
MOMENTS = [1, 2]
"""Default moments to compute: 1=mean, 2=standard deviation."""

MOMENT_WEIGHTS = {1: 0.9, 2: 0.1}
"""Default moment weighting (None = equal weights for all moments)."""

EPS = 1e-6
"""Default epsilon for numerical stability in divisions."""

# Patch weighting parameters
PATCH_WEIGHTING = 'scaled'
"""Default patch weighting scheme: 'scaled', 'energy', 'mean', or 'none'."""

PATCH_WEIGHT_MIN = 0.25
"""Default minimum weight for 'scaled' weighting mode."""

PATCH_WEIGHT_MAX = 1.0
"""Default maximum weight for 'scaled' weighting mode."""

# Patch masking parameters
MAX_PATCH_MASKED = 0
"""Default threshold for masking low-activity patches (0 = only zero patches)."""

PATHOLOGICAL_PENALTY = 10000.0
"""Penalty value returned when predictions are pathological (NaN, negative mean, or all masked).
Should be much larger than typical metric values to signal poor trial performance to Ray Tune."""

# Normalization parameters
USE_POISSON_NORMALIZATION = True
"""Default: use physics-informed Poisson normalization (PET-optimized)."""

SCALE = 'mean'
"""Default scale for generic mode normalization: 'mean' or 'std'."""

# ============================
# Hybrid Loss Parameters
# ============================

HYBRID_EPSILON = 1e-8
"""Default epsilon for HybridLoss gradient normalization."""

HYBRID_SHOW_COMPONENTS = False
"""Default: print HybridLoss component diagnostics during training."""

HYBRID_C_MOMENTUM = 0.05
"""Default EMA momentum for HybridLoss gradient scale coefficient C."""

# ============================
# VarWeightedMSE Parameters
# ============================

COUNTS_PER_BQ = 60.0
"""Default photon counts per bequerel for variance-weighted MSE.
For PET activity maps, use ~60. For annihilation maps, use 1."""

VAR_WEIGHTED_EPSILON = 1e-8
"""Default epsilon for VarWeightedMSE denominator (prevents division by zero)."""
