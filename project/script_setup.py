# -*- coding: utf-8 -*-
"""
Environment setup and helper functions for FlexCNN training/tuning/testing scripts.
This module provides reusable setup code for both Jupyter notebooks and standalone scripts.
"""

import os
import sys
import glob
import importlib
import inspect
import types
import subprocess
import pkgutil


def sense_colab():
    """Detect if running in Google Colab."""
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    return IN_COLAB


def sense_device(device='sense'):
    """Detect and set device (cuda, cpu)."""
    if device == 'sense':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    elif device == 'cpu':
        device = 'cpu'
    elif device == 'cuda':
        device = 'cuda'
    return device


def install_packages(IN_COLAB=True, force_reinstall=False, include_optional=True, ray_version=None):
    """
    Installs required Python packages efficiently.
    - Detects if running in Colab or locally.
    - Installs missing packages only (unless force_reinstall=True).
    - For local: always installs CUDA-enabled PyTorch (cu124).
    - Optionally pin Ray version with ray_version (e.g., "2.9.0").
    """

    # Base list of non-PyTorch packages
    other_packages = [
        "ray[tune]", "tensorboardX", "hyperopt", "optuna",
        "numpy", "pandas", "matplotlib",
        "scikit-image", "scipy"
    ]

    # Optional packages
    optional_packages = ["tensorboard"]
    widgets_packages = ["ipywidgets"]

    missing = []

    # On Colab, just use standard installation
    if IN_COLAB:
        packages = [
            "torch", "torchvision", "torchaudio",
            "ray[tune]", "tensorboardX", "hyperopt", "optuna",
            "numpy", "pandas", "matplotlib",
            "scikit-image", "scipy"
        ]
        optional_packages_to_install = ["tensorboard"] if include_optional else []
        widgets_packages_to_install = ["ipywidgets"] if include_optional else []
        
        for pkg in packages + optional_packages_to_install + widgets_packages_to_install:
            pkg_name = pkg.split("[")[0]
            if pkg_name == "ray":
                try:
                    import ray
                    import ray.tune
                    ray_tune_installed = True
                except ImportError:
                    ray_tune_installed = False
                
                # Pin Ray version if specified
                if ray_version:
                    pkg = f"ray[tune]=={ray_version}"
                
                if force_reinstall or not ray_tune_installed:
                    missing.append(pkg)
            elif importlib.util.find_spec(pkg_name) is None or force_reinstall:
                missing.append(pkg)
        
        if not missing:
            print("✅ All required packages already installed.")
            return
        
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        
        # For Colab, install PyTorch with CUDA support (cu124 works on Colab)
        torch_packages = [p for p in missing if p.split("[")[0] in ["torch", "torchvision", "torchaudio"]]
        other_missing = [p for p in missing if p.split("[")[0] not in ["torch", "torchvision", "torchaudio"]]
        
        if torch_packages:
            print(f"📦 Installing PyTorch with CUDA (cu124) for Colab GPU support...")
            cmd_torch = [sys.executable, "-m", "pip", "install", "--upgrade", "--index-url", "https://download.pytorch.org/whl/cu124"] + torch_packages
            try:
                subprocess.check_call(cmd_torch)
                print("✅ PyTorch installation complete.")
            except subprocess.CalledProcessError as e:
                print(f"❌ PyTorch installation failed: {e}")
                return
        
        if other_missing:
            print(f"📦 Installing other packages...")
            try:
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + other_missing
                subprocess.check_call(cmd)
                print("✅ Installation complete.")
            except subprocess.CalledProcessError as e:
                print(f"❌ Installation failed: {e}")
                print("🔁 Retrying installs individually (no cache)...")
                failed = []
                for pkg in other_missing:
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", pkg])
                    except subprocess.CalledProcessError:
                        failed.append(pkg)
                if failed:
                    raise RuntimeError(f"Failed to install packages: {failed}")

        # Validate Ray is available before importing package modules
        try:
            import ray  # noqa: F401
            import ray.tune  # noqa: F401
        except ImportError as e:
            raise RuntimeError("Ray Tune is required but not installed. Please re-run the install cell.") from e
        return

    # Local: Always install CUDA PyTorch (cu124), other packages with standard PyPI
    print("🖥️  Local environment detected. Installing CUDA-enabled PyTorch...")
    
    torch_packages = ["torch", "torchvision", "torchaudio"]
    
    # Check which packages are missing
    for pkg in other_packages:
        pkg_name = pkg.split("[")[0]
        if pkg_name == "ray":
            try:
                import ray
                import ray.tune
                ray_tune_installed = True
            except ImportError:
                ray_tune_installed = False
            
            # Pin Ray version if specified
            if ray_version:
                pkg = f"ray[tune]=={ray_version}"
            
            if force_reinstall or not ray_tune_installed:
                missing.append(pkg)
        elif importlib.util.find_spec(pkg_name) is None or force_reinstall:
            missing.append(pkg)
    
    if include_optional:
        missing += optional_packages + widgets_packages
    
    missing = list(dict.fromkeys(missing))
    
    # Install torch with CUDA index
    print(f"📦 Installing PyTorch with CUDA (cu124)...")
    cmd_torch = [sys.executable, "-m", "pip", "install", "--upgrade", "--index-url", "https://download.pytorch.org/whl/cu124"] + torch_packages
    try:
        subprocess.check_call(cmd_torch)
        print("✅ PyTorch installation complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")
        return
    
    # Install other packages with standard PyPI
    if missing:
        print(f"📦 Installing other packages...")
        cmd_other = [sys.executable, "-m", "pip", "install", "--upgrade"] + missing
        try:
            subprocess.check_call(cmd_other)
            print("✅ Other packages installation complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Other packages installation failed: {e}")
    
    # Diagnose CUDA
    print("\n" + "="*60)
    print("CUDA Diagnostic Information:")
    print("="*60)
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ CUDA is NOT available - this is a problem!")
            print("   Checking nvidia-smi...")
            try:
                result = subprocess.check_output("nvidia-smi", shell=True).decode()
                print("   nvidia-smi output:")
                for line in result.split('\n')[:10]:
                    print(f"     {line}")
            except Exception as e:
                print(f"   nvidia-smi not found: {e}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
    print("="*60 + "\n")


def reload_submodules(pkg):
    """Reload all submodules in a package to pick up code changes."""
    for importer, modname, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            sub_module = importlib.import_module(modname)
            importlib.reload(sub_module)
        except Exception:
            pass


def reload_package(repo_name: str = "FlexCNN_for_Medical_Physics", verbose: bool = True):
    """
    Reload a package and inject all symbols into caller's globals.
    
    Useful in Jupyter notebooks when you've edited package code and want to
    reload changes without restarting the kernel.
    
    Args:
        repo_name: Name of the package to reload
        verbose: Print status messages
    
    Example:
        # At top of notebook cell after editing package code
        reload_package()
        # Now all updated functions/classes are available
    """
    if verbose:
        print(f"🔄 Reloading {repo_name} package...")
    
    # Import or reload the main package
    if repo_name in sys.modules:
        package = importlib.reload(sys.modules[repo_name])
    else:
        package = importlib.import_module(repo_name)
    
    # Reload all submodules
    reload_submodules(package)
    
    # Gather all symbols from all modules
    imported = {}
    for _, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            mod = importlib.import_module(modname)
            imported.update({name: obj for name, obj in vars(mod).items() if not name.startswith('_')})
        except Exception:
            pass
    
    # Inject symbols into caller's globals
    if verbose:
        print("✨ Injecting all symbols into global namespace...")
    caller_globals = inspect.stack()[1].frame.f_globals
    caller_globals.update(imported)
    if verbose:
        print(f"✅ Reload complete: {len(imported)} symbols updated.")


def resolve_repo_root():
    """Resolve repo root by searching for setup.py/pyproject.toml."""
    try:
        start_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        start_dir = os.getcwd()

    repo_root = start_dir
    while repo_root != os.path.dirname(repo_root):
        if os.path.exists(os.path.join(repo_root, "setup.py")) or os.path.exists(os.path.join(repo_root, "pyproject.toml")):
            return repo_root
        repo_root = os.path.dirname(repo_root)

    raise FileNotFoundError("Could not locate repo root (setup.py or pyproject.toml).")


def setup_colab_environment(
    github_username: str = "peterlabcl8",
    repo_name: str = "FlexCNN_for_Medical_Physics",
    local_repo_path: str = None,
    skip_git_update: bool = False,
    force_fresh_clone: bool = False,
    verbose: bool = True):
    """
    Setup environment for Colab: clone/pull repo and install via pip.
    Injects all package symbols into caller's globals.
    
    Args:
        github_username: GitHub username for the repository
        repo_name: Repository name
        local_repo_path: Local path (unused for Colab, kept for consistency)
        skip_git_update: If True, skip git pull (useful if already up-to-date or if git operations fail)
        force_fresh_clone: If True, remove existing repo and clone fresh from GitHub
        verbose: Print status messages
    """
    import shutil
    
    # Determine base directory
    base_dir = "/content"
    repo_path = os.path.join(base_dir, repo_name)
    repo_url = f"https://github.com/{github_username}/{repo_name}.git"

    # Remove old clone if force_fresh_clone is True
    if force_fresh_clone and os.path.exists(repo_path):
        if verbose:
            print(f"🗑️  Removing old clone: {repo_path}")
        shutil.rmtree(repo_path)

    # Clone or update
    if not os.path.exists(repo_path):
        if verbose:
            print(f"📦 Cloning {repo_name} into {base_dir}...")
        try:
            subprocess.run(["git", "clone", repo_url], cwd=base_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git clone failed: {e}")
            print(f"   Proceeding without updating repository...")
    elif not skip_git_update:
        if verbose:
            print(f"🔄 Pulling latest changes in {repo_path}...")
        try:
            subprocess.run(["git", "pull"], cwd=repo_path, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git pull failed: {e}")
            print(f"   Proceeding with existing repository...")
    else:
        if verbose:
            print(f"⏭️  Skipping git update (skip_git_update=True)")

    # Install package in editable mode
    if verbose:
        print("⚙️ Installing the package in editable mode...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
                       cwd=repo_path, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Package installation failed: {e}")
        print(f"   Attempting to proceed anyway...")

    # Ensure repo path is importable
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    # Import the package
    package = importlib.import_module(repo_name)

    # Reload all submodules
    reload_submodules(package)

    # Gather all symbols
    imported = {}
    for _, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        mod = importlib.import_module(modname)
        for name, obj in inspect.getmembers(mod):
            if not name.startswith("_"):
                imported[name] = obj

    # Inject symbols into caller's globals
    if verbose:
        print("✨ Injecting all symbols into global namespace...")
    caller_globals = inspect.stack()[1].frame.f_globals
    caller_globals.update(imported)
    if verbose:
        print(f"✅ Setup complete: {len(imported)} symbols loaded into globals.")


def setup_local_environment(
    repo_name: str = "FlexCNN_for_Medical_Physics",
    mode: str = "walk",
    verbose: bool = True):
    """
    Setup environment for local machine: install or walk package.
    Injects all package symbols into caller's globals.
    """
    if mode not in ("walk", "install"):
        raise ValueError(f"setup_mode must be 'walk' or 'install', got '{mode}'.")

    package_root = resolve_repo_root()

    if mode == "install":
        if verbose:
            print(f"⚙️ Installing the package in editable mode from {package_root}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
                           cwd=package_root, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Package installation failed: {e}")
            print("   Attempting to proceed anyway...")
    
    # Add to sys.path
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
        if verbose:
            print(f"📂 Added {package_root} to sys.path")

    # Import and walk the package
    if verbose:
        print(f"📦 Loading {repo_name} package ({mode} mode)...")
    package = importlib.import_module(repo_name)
    
    # Reload all submodules to pick up code changes
    reload_submodules(package)
    
    # Gather all symbols from all modules
    imported = {}
    for _, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            mod = importlib.import_module(modname)
            imported.update({name: obj for name, obj in vars(mod).items() if not name.startswith('_')})
        except Exception:
            pass
    
    # Inject symbols into caller's globals
    if verbose:
        print("✨ Injecting all symbols into global namespace...")
    caller_globals = inspect.stack()[1].frame.f_globals
    caller_globals.update(imported)
    if verbose:
        print(f"✅ Setup complete: {len(imported)} symbols loaded into globals.")


def refresh_repo(
    IN_COLAB = True,
    repo_name: str = "FlexCNN_for_Medical_Physics",
    github_username: str = "petercl8",
    local_repo_path: str = None,
    force_fresh_clone: bool = False,
    auto_import: bool = True,
    verbose: bool = True):
    """
    Clone/pull and install the repo, then optionally auto-import all modules.
    Also reloads all submodules to reflect changes without restarting the runtime.
    
    Args:
        IN_COLAB: Whether running in Colab
        repo_name: Repository name
        github_username: GitHub username
        local_repo_path: Local path (required if not in Colab)
        force_fresh_clone: If True, remove existing repo and clone fresh
        auto_import: If True, inject all symbols into caller's globals
        verbose: Print status messages
    """
    import shutil
    
    # --- Determine base directory ---
    base_dir = "/content" if IN_COLAB else local_repo_path
    if base_dir is None:
        raise ValueError("local_repo_path must be provided if not in Colab")

    repo_path = os.path.join(base_dir, repo_name)
    repo_url = (
        f"https://github.com/{github_username}/{repo_name}.git"
        if IN_COLAB
        else f"git@github.com:{github_username}/{repo_name}.git"
    )

    # --- Remove old clone if force_fresh_clone is True ---
    if force_fresh_clone and os.path.exists(repo_path):
        if verbose:
            print(f"🗑️  Removing old clone: {repo_path}")
        shutil.rmtree(repo_path)

    # --- Clone or update ---
    if not os.path.exists(repo_path):
        if verbose:
            print(f"📦 Cloning {repo_name} into {base_dir}...")
        subprocess.run(["git", "clone", repo_url], cwd=base_dir, check=True)
    else:
        if verbose:
            print(f"🔄 Pulling latest changes in {repo_path}...")
        subprocess.run(["git", "pull"], cwd=repo_path, check=True)

    # --- Install package in editable mode ---
    if verbose:
        print("⚙️ Installing the package in editable mode...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
                   cwd=repo_path, check=True)

    # --- Ensure repo path is importable ---
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    # --- Import the package ---
    package = importlib.import_module(repo_name)

    # --- Reload all submodules recursively ---
    def reload_submodules(pkg):
        for _, modname, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if modname in sys.modules:
                importlib.reload(sys.modules[modname])
            else:
                importlib.import_module(modname)
        importlib.reload(pkg)

    reload_submodules(package)

    # --- Gather all symbols ---
    imported = {}
    for _, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        mod = importlib.import_module(modname)
        for name, obj in inspect.getmembers(mod):
            if not name.startswith("_"):
                imported[name] = obj

    # --- Inject symbols into caller's globals if requested ---
    if auto_import:
        if verbose:
            print("✨ Injecting all symbols into global namespace...")
        caller_globals = inspect.stack()[1].frame.f_globals
        caller_globals.update(imported)
        if verbose:
            print(f"✅ Setup complete: {len(imported)} symbols loaded into globals.")
    else:
        if verbose:
            print(f"✅ Imported {len(imported)} symbols (not injected).")
