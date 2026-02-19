# -*- coding: utf-8 -*-
"""
FlexCNN Medical Physics Pipeline - Main Runner Script

This script performs tuning, training, testing, or visualization of CNNs
for PET image reconstruction from sinograms.

Usage:
    python run_pipeline.py
To customize parameters, edit user_params.py before running this script.
"""
skip_package_installation = True

import os
import sys

# Import setup functions
from FlexCNN_for_Medical_Physics.functions.helper.display_images import configure_plotting
from FlexCNN_for_Medical_Physics.functions.setup_environment.list_compute_resources import setup_project_dirs
from script_setup import (
    sense_colab, sense_device, install_packages,
    setup_colab_environment, setup_local_environment
)
from FlexCNN_for_Medical_Physics.functions.main_run_functions.pipeline import run_pipeline

# --- Sense environment ---
IN_COLAB = sense_colab()

# Import build_dicts after environment setup (depends on package being importable)
from build_dicts import build_all_dicts
# Import user parameters (no package dependencies)
from USER_PARAMS import get_params

# --- Get user parameters ---
params = get_params()

# --- Setup Environment ---
# Three setup modes:
#   1. Colab: Install packages + clone/pull repo + walks modules
#   2. Local + walk: Reload existing modules (fast, assumes packages installed)
#   3. Local + install: Install packages via pip + set up package locally

if not skip_package_installation:
    print("Installing packages...")
    install_packages(IN_COLAB, ray_version=params['ray_tune_version'])
else:
    print("⚠️ Skipping package installation as per user parameters.")

if IN_COLAB:
    setup_colab_environment(
        github_username=params['github_username'],
        repo_name=params['repo_name'],
        skip_git_update=params['skip_colab_git_update']
    )
else:
    # Setup local environment using specified mode (walk or install)
    setup_local_environment(
        repo_name=params['repo_name'],
        mode=params['setup_mode_type']
    )

# --- Test Resources ---
list_compute_resources()

# --- Configure Plotting ---
configure_plotting(plot_mode=params['plot_mode'])

# --- Set main project directory ---
params['project_dirPath'] = setup_project_dirs(
    IN_COLAB,
    params['project_local_dirPath'],
    params['project_colab_dirPath'],
    mount_colab_drive=False
)

# --- Set Device ---
params['device_opt'] = sense_device(device=params['device_opt'])

# --- Build all dictionaries ---
print("🔧 Building configuration dictionaries...")
all_dicts = build_all_dicts(params)

# Extract main dictionaries
config = all_dicts['config']
paths = all_dicts['paths']
settings = all_dicts['settings']
base_dirs = all_dicts['base_dirs']
tune_opts = all_dicts['tune_opts']
test_opts = all_dicts['test_opts']

#from pprint import pprint
#pprint(config)

# --- Run Pipeline ---
print(f"🚀 Running pipeline in '{params['run_mode']}' mode...")
run_pipeline(
    config=config,
    paths=paths,
    settings=settings,
    tune_opts=tune_opts,
    base_dirs=base_dirs,
    test_opts=test_opts,
)

print("✅ Pipeline execution complete!")