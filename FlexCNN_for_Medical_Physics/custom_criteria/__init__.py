"""
Custom losses and metrics for medical image reconstruction.

This module provides a unified interface for all loss functions and evaluation metrics,
with default parameter values centralized in defaults.py for easy configuration.

Public API
----------
Losses:
    HybridLoss : Dynamically weighted combination of base and stats losses
    PatchwiseMomentLoss : Patchwise statistical moment matching loss
    VarWeightedMSE : Variance-weighted MSE for Poisson data

Metrics:
    patchwise_moment_metric : Evaluation metric for patchwise moment matching
    custom_metric : Simple wrapper for patchwise_moment_metric

Configuration:
    defaults : Module containing all default parameter values

Examples
--------
>>> from FlexCNN_for_Medical_Physics.custom_criteria import PatchwiseMomentLoss, patchwise_moment_metric
>>> import torch
>>> 
>>> # Create loss with defaults
>>> loss_fn = PatchwiseMomentLoss()
>>> 
>>> # Evaluate with metric
>>> pred = torch.randn(4, 1, 128, 128)
>>> target = torch.randn(4, 1, 128, 128)
>>> loss = loss_fn(pred, target)
>>> metric = patchwise_moment_metric(pred, target)
"""

# Import all public classes and functions
from .losses import HybridLoss, PatchwiseMomentLoss, VarWeightedMSE
from .metrics import patchwise_moment_metric, custom_metric
from . import defaults

# Define public API
__all__ = [
    # Loss classes
    'HybridLoss',
    'PatchwiseMomentLoss',
    'VarWeightedMSE',
    # Metric functions
    'patchwise_moment_metric',
    'custom_metric',
    # Configuration
    'defaults',
]
