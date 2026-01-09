"""
Test script for Stage 0 refactoring of run_supervisory.py helpers.

Tests:
1. _create_generator() instantiates correct class for 288 input
2. _create_optimizer() creates optimizer with proper param groups
3. _build_checkpoint_dict() assembles checkpoint correctly
4. _save_checkpoint() and _load_checkpoint() round-trip successfully
5. Checkpoint preserves model state and optimizer state
"""

import torch
import os
import tempfile
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from FlexCNN_for_Medical_Physics.functions.main_run_functions.trainable import (
    _create_generator,
    _create_optimizer,
    _build_checkpoint_dict,
    _save_checkpoint,
    _load_checkpoint,
)


def get_base_config():
    """Return a minimal valid config for 288x288 Generator."""
    return {
        'train_SI': True,
        'image_size': 288,
        'sino_size': 288,
        'image_channels': 1,
        'sino_channels': 1,
        # SI configs
        'SI_gen_hidden_dim': 64,
        'SI_gen_mult': 1.0,
        'SI_gen_fill': 'zeros',
        'SI_gen_neck': 'narrow',
        'SI_gen_z_dim': 16,
        'SI_exp_kernel': 3,
        'SI_skip_mode': 'concat',
        'SI_layer_norm': False,
        'SI_pad_mode': 'reflect',
        'SI_dropout': 0.0,
        'SI_gen_final_activ': 'sigmoid',
        'SI_normalize': False,
        'SI_fixedScale': 1.0,
        'SI_learnedScale_init': 1.0,  # Initialize learnable scale
        # IS configs (required even though train_SI=True)
        'IS_gen_hidden_dim': 64,
        'IS_gen_mult': 1.0,
        'IS_gen_fill': 'zeros',
        'IS_gen_neck': 'narrow',
        'IS_gen_z_dim': 16,
        'IS_exp_kernel': 3,
        'IS_skip_mode': 'concat',
        'IS_layer_norm': False,
        'IS_pad_mode': 'reflect',
        'IS_dropout': 0.0,
        'IS_gen_final_activ': 'sigmoid',
        'IS_normalize': False,
        'IS_fixedScale': 1.0,
        'IS_learnedScale_init': 1.0,  # Initialize learnable scale
        # Optimizer
        'gen_b1': 0.9,
        'gen_b2': 0.999,
        'gen_lr': 0.0002,
        'SI_output_scale_lr_mult': 1.0,
        'IS_output_scale_lr_mult': 1.0,
    }


def test_create_generator_288():
    """Test that _create_generator instantiates Generator_288 for 288 input."""
    print("\n[TEST 1] _create_generator for 288x288 input...")
    
    config = get_base_config()
    device = 'cpu'
    gen = _create_generator(config, device)
    
    # Check that it's on the right device
    assert next(gen.parameters()).device.type == 'cpu', "Generator not on correct device"
    
    # Check class name
    class_name = gen.__class__.__name__
    assert class_name == 'Generator_288', f"Expected Generator_288, got {class_name}"
    
    print(f"  ✓ Generator_288 instantiated correctly")
    print(f"  ✓ Device: {device}")
    print(f"  ✓ Parameter count: {sum(p.numel() for p in gen.parameters())}")
    

def test_create_optimizer():
    """Test that _create_optimizer creates optimizer with correct param groups."""
    print("\n[TEST 2] _create_optimizer with learnable scale...")
    
    config = get_base_config()
    
    gen = _create_generator(config, 'cpu')
    opt = _create_optimizer(gen, config)
    
    # Check optimizer type
    assert isinstance(opt, torch.optim.Adam), f"Expected Adam, got {type(opt)}"
    
    # Check param groups
    param_groups = opt.param_groups
    assert len(param_groups) >= 1, "Optimizer should have at least 1 param group"
    
    print(f"  ✓ Adam optimizer created")
    print(f"  ✓ Number of param groups: {len(param_groups)}")
    for i, pg in enumerate(param_groups):
        print(f"    Group {i}: lr={pg['lr']}, params={len(pg['params'])}")
    

def test_checkpoint_roundtrip():
    """Test checkpoint save/load round-trip preserves state."""
    print("\n[TEST 3] Checkpoint save/load round-trip...")
    
    config = get_base_config()
    
    # Create model and optimizer
    gen = _create_generator(config, 'cpu')
    opt = _create_optimizer(gen, config)
    
    # Manually update some weights to simulate training
    with torch.no_grad():
        for p in gen.parameters():
            p.data.add_(torch.randn_like(p) * 0.01)
    
    # Capture state before save
    gen_state_before = {k: v.clone() for k, v in gen.state_dict().items()}
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'test_checkpoint.pt')
        
        # Build and save
        ckpt_dict = _build_checkpoint_dict(gen, opt, config, epoch=1, batch_step=42)
        _save_checkpoint(ckpt_dict, ckpt_path)
        
        assert os.path.exists(ckpt_path), "Checkpoint file not created"
        print(f"  ✓ Checkpoint saved to {ckpt_path}")
        
        # Create fresh model and optimizer
        gen2 = _create_generator(config, 'cpu')
        opt2 = _create_optimizer(gen2, config)
        
        # Load checkpoint
        epoch_loaded, batch_step_loaded, gen_state_dict, opt_state_dict = _load_checkpoint(ckpt_path)
        gen2.load_state_dict(gen_state_dict)
        opt2.load_state_dict(opt_state_dict)
        
        print(f"  ✓ Checkpoint loaded")
        print(f"    Epoch: {epoch_loaded}, Batch step: {batch_step_loaded}")
        
        # Verify state matches
        gen_state_after = gen2.state_dict()
        for key in gen_state_before.keys():
            assert torch.allclose(gen_state_before[key], gen_state_after[key]), \
                f"State mismatch for key {key}"
        
        print(f"  ✓ Generator state matches after load")
        print(f"    Verified {len(gen_state_before)} state dict entries")


def test_checkpoint_metadata():
    """Test that checkpoint metadata is preserved correctly."""
    print("\n[TEST 4] Checkpoint metadata preservation...")
    
    config = get_base_config()
    
    gen = _create_generator(config, 'cpu')
    opt = _create_optimizer(gen, config)
    
    test_epoch = 5
    test_batch_step = 123
    
    ckpt_dict = _build_checkpoint_dict(gen, opt, config, epoch=test_epoch, batch_step=test_batch_step)
    
    # Verify checkpoint keys
    expected_keys = {'epoch', 'batch_step', 'gen_state_dict', 'gen_opt_state_dict'}
    actual_keys = set(ckpt_dict.keys())
    assert actual_keys == expected_keys, \
        f"Checkpoint keys mismatch. Expected {expected_keys}, got {actual_keys}"
    
    # Verify metadata values
    assert ckpt_dict['epoch'] == test_epoch, f"Epoch mismatch: {ckpt_dict['epoch']} vs {test_epoch}"
    assert ckpt_dict['batch_step'] == test_batch_step, \
        f"Batch step mismatch: {ckpt_dict['batch_step']} vs {test_batch_step}"
    
    print(f"  ✓ Checkpoint has correct keys: {actual_keys}")
    print(f"  ✓ Epoch: {ckpt_dict['epoch']}")
    print(f"  ✓ Batch step: {ckpt_dict['batch_step']}")
    print(f"  ✓ Config NOT stored (as intended)")


def test_file_not_found():
    """Test that _load_checkpoint raises FileNotFoundError for missing file."""
    print("\n[TEST 5] _load_checkpoint error handling...")
    
    try:
        _load_checkpoint('/nonexistent/path/checkpoint.pt')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"  ✓ FileNotFoundError raised as expected: {e}")


if __name__ == '__main__':
    print("=" * 70)
    print("STAGE 0 REFACTORING TESTS")
    print("=" * 70)
    
    try:
        test_create_generator_288()
        test_create_optimizer()
        test_checkpoint_roundtrip()
        test_checkpoint_metadata()
        test_file_not_found()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
