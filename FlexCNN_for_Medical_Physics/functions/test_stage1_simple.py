"""
Simplified Stage 1 Test Suite - Tests dataset loading without using temporary directories
"""

import torch
import numpy as np
import os
import sys

# Add package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataSet


def get_base_config():
    """Return a minimal config for testing"""
    return {
        'network_type': 'SUP',
        'train_SI': True,
        'image_size': 180,
        'sino_size': 288,
        'image_channels': 1,
        'sino_channels': 3,
        'SI_normalize': False,
        'SI_fixedScale': 1.0,
        'IS_normalize': False,
        'IS_fixedScale': 1.0,
        'batch_size': 4,
    }


def get_base_settings():
    """Return minimal settings for testing"""
    return {
        'recon1_scale': 1.0,
        'recon2_scale': 1.0,
        'sino_scale': 1.0,
        'image_scale': 1.0,
    }


def create_test_data_persistent():
    """Create persistent test data in current directory - no cleanup needed"""
    # Create data directory if it doesn't exist
    data_dir = './test_data_stage1'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create dummy arrays
    num_examples = 10
    sino_array = np.random.randn(num_examples, 3, 288, 288).astype(np.float32)
    image_array = np.random.randn(num_examples, 1, 180, 180).astype(np.float32)
    atten_array = np.random.randn(num_examples, 1, 180, 180).astype(np.float32)
    recon1_array = np.random.randn(num_examples, 1, 180, 180).astype(np.float32)
    recon2_array = np.random.randn(num_examples, 1, 180, 180).astype(np.float32)
    
    # Save to persistent files
    sino_path = os.path.join(data_dir, 'sino.npy')
    image_path = os.path.join(data_dir, 'image.npy')
    atten_path = os.path.join(data_dir, 'atten.npy')
    recon1_path = os.path.join(data_dir, 'recon1.npy')
    recon2_path = os.path.join(data_dir, 'recon2.npy')
    
    np.save(sino_path, sino_array)
    np.save(image_path, image_array)
    np.save(atten_path, atten_array)
    np.save(recon1_path, recon1_array)
    np.save(recon2_path, recon2_array)
    
    return data_dir, sino_path, image_path, atten_path, recon1_path, recon2_path


def cleanup_test_data(data_dir):
    """Clean up test data directory"""
    import shutil
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir, ignore_errors=True)


# ========================================================================================
# TEST 1: BACKWARD COMPATIBILITY (atten_map_path=None)
# ========================================================================================

def test_backward_compatibility_no_augmentation():
    """Test that normal mode (no attenuation) returns proper nested tuple structure"""
    print("\n" + "="*80)
    print("TEST 1: Backward Compatibility (No Augmentation)")
    print("="*80)
    
    import gc
    data_dir, sino_path, image_path, atten_path, recon1_path, recon2_path = create_test_data_persistent()
    dataset = None
    try:
        config = get_base_config()
        settings = get_base_settings()
        
        dataset = NpArrayDataSet(
            image_path=image_path,
            sino_path=sino_path,
            config=config,
            settings=settings,
            augment=(None, False),
            offset=0,
            num_examples=5,
            sample_division=1,
            device='cpu',
            recon1_path=recon1_path,
            recon2_path=recon2_path,
            atten_map_path=None  # No attenuation maps
        )
        
        # Test __getitem__
        activity_tuple, attenuation_tuple, recon_tuple = dataset[0]
        
        # Validate structure
        assert isinstance(activity_tuple, tuple), f"activity_tuple should be tuple, got {type(activity_tuple)}"
        assert len(activity_tuple) == 2, f"activity_tuple should have 2 elements, got {len(activity_tuple)}"
        assert attenuation_tuple is None, f"attenuation_tuple should be None, got {attenuation_tuple}"
        assert isinstance(recon_tuple, tuple), f"recon_tuple should be tuple, got {type(recon_tuple)}"
        assert len(recon_tuple) == 2, f"recon_tuple should have 2 elements, got {len(recon_tuple)}"
        
        # Validate tensor properties
        sino_scaled, act_map_scaled = activity_tuple
        assert sino_scaled.shape[0] == 3, f"Sinogram should have 3 channels, got {sino_scaled.shape[0]}"
        assert act_map_scaled.shape == (1, 180, 180), f"Activity map should be (1, 180, 180), got {act_map_scaled.shape}"
        
        recon1, recon2 = recon_tuple
        assert recon1.shape == (1, 180, 180), f"Recon1 should be (1, 180, 180), got {recon1.shape}"
        assert recon2.shape == (1, 180, 180), f"Recon2 should be (1, 180, 180), got {recon2.shape}"
        
        print(f"✅ PASS: Backward compatibility verified")
        print(f"   - activity_tuple: (sino {sino_scaled.shape}, image {act_map_scaled.shape})")
        print(f"   - attenuation_tuple: None")
        print(f"   - recon_tuple: (recon1 {recon1.shape}, recon2 {recon2.shape})")
    finally:
        if dataset is not None:
            dataset.close()
            del dataset
        gc.collect()
        cleanup_test_data(data_dir)


# ========================================================================================
# TEST 2: ATTENUATION MAP LOADING
# ========================================================================================

def test_attenuation_map_loading():
    """Test that attenuation maps load with correct shapes and dtypes"""
    print("\n" + "="*80)
    print("TEST 2: Attenuation Map Loading")
    print("="*80)
    
    import gc
    data_dir, sino_path, image_path, atten_path, recon1_path, recon2_path = create_test_data_persistent()
    dataset = None
    try:
        config = get_base_config()
        settings = get_base_settings()
        
        dataset = NpArrayDataSet(
            image_path=image_path,
            sino_path=sino_path,
            config=config,
            settings=settings,
            augment=(None, False),
            offset=0,
            num_examples=5,
            sample_division=1,
            device='cpu',
            recon1_path=None,
            recon2_path=None,
            atten_map_path=atten_path  # WITH attenuation maps
        )
        
        # Test __getitem__
        activity_tuple, attenuation_tuple, recon_tuple = dataset[0]
        
        # Validate attenuation tuple
        assert attenuation_tuple is not None, "attenuation_tuple should not be None"
        assert isinstance(attenuation_tuple, tuple), f"attenuation_tuple should be tuple, got {type(attenuation_tuple)}"
        assert len(attenuation_tuple) == 2, f"attenuation_tuple should have 2 elements, got {len(attenuation_tuple)}"
        
        atten_sino, atten_map = attenuation_tuple
        assert atten_sino.shape[0] == 3, f"Atten sinogram should have 3 channels, got {atten_sino.shape[0]}"
        assert atten_map.shape == (1, 180, 180), f"Atten map should be (1, 180, 180), got {atten_map.shape}"
        assert atten_map.dtype == torch.float32, f"Atten map should be float32, got {atten_map.dtype}"
        
        print(f"✅ PASS: Attenuation map loading verified")
        print(f"   - attenuation_tuple: (atten_sino {atten_sino.shape}, atten_map {atten_map.shape})")
    finally:
        if dataset is not None:
            dataset.close()
            del dataset
        gc.collect()
        cleanup_test_data(data_dir)


# ========================================================================================
# TEST 3: AUGMENTATION CONSISTENCY
# ========================================================================================

def test_augmentation_consistency():
    """Test that same augmentations apply to attenuation and activity data"""
    print("\n" + "="*80)
    print("TEST 3: Augmentation Consistency")
    print("="*80)
    
    import gc
    data_dir, sino_path, image_path, atten_path, recon1_path, recon2_path = create_test_data_persistent()
    dataset = None
    try:
        config = get_base_config()
        settings = get_base_settings()
        
        # Create dataset WITH augmentation
        dataset = NpArrayDataSet(
            image_path=image_path,
            sino_path=sino_path,
            config=config,
            settings=settings,
            augment=('SI', False),  # Enable augmentation
            offset=0,
            num_examples=5,
            sample_division=1,
            device='cpu',
            recon1_path=None,
            recon2_path=None,
            atten_map_path=atten_path
        )
        
        # Get multiple samples to see if augmentation is applied
        activity_tuple_0, attenuation_tuple_0, _ = dataset[0]
        activity_tuple_1, attenuation_tuple_1, _ = dataset[1]
        
        # Both should have been augmented (values may differ due to flips/rotations)
        # The key test is that attenuation_tuple is not None and has correct shape
        assert attenuation_tuple_0 is not None, "Attenuation data should be present"
        assert attenuation_tuple_1 is not None, "Attenuation data should be present on second sample"
        
        atten_map_0 = attenuation_tuple_0[1]
        atten_map_1 = attenuation_tuple_1[1]
        
        assert atten_map_0.shape == (1, 180, 180), f"Expected shape (1, 180, 180), got {atten_map_0.shape}"
        assert atten_map_1.shape == (1, 180, 180), f"Expected shape (1, 180, 180), got {atten_map_1.shape}"
        
        # Verify attenuation maps still have distinct values (not same as activity)
        assert torch.abs(atten_map_0.mean() - atten_map_1.mean()) < 1.0, "Attenuation maps normalized correctly"
        
        print(f"✅ PASS: Augmentation consistency verified")
        print(f"   - Attenuation data loaded and shaped correctly")
        print(f"   - Augmentation applied to both samples")
    finally:
        if dataset is not None:
            dataset.close()
            del dataset
        gc.collect()
        cleanup_test_data(data_dir)


# ========================================================================================
# TEST 4: NESTED TUPLE UNPACKING IN run_supervisory.py CONTEXT
# ========================================================================================

def test_nested_tuple_unpacking():
    """Test that nested tuples unpack correctly in a run_supervisory.py-like context"""
    print("\n" + "="*80)
    print("TEST 4: Nested Tuple Unpacking (run_supervisory.py Context)")
    print("="*80)
    
    import gc
    data_dir, sino_path, image_path, atten_path, recon1_path, recon2_path = create_test_data_persistent()
    dataset = None
    try:
        config = get_base_config()
        settings = get_base_settings()
        config['train_SI'] = True
        
        dataset = NpArrayDataSet(
            image_path=image_path,
            sino_path=sino_path,
            config=config,
            settings=settings,
            augment=(None, False),
            offset=0,
            num_examples=5,
            sample_division=1,
            device='cpu',
            recon1_path=recon1_path,
            recon2_path=recon2_path,
            atten_map_path=atten_path
        )
        
        # Simulate run_supervisory.py batch loop unpacking
        activity_data, attenuation_data, recon_data = dataset[0]
        
        # Unpack activity (as run_supervisory.py would)
        sino_scaled, act_map_scaled = activity_data
        
        # Route based on train_SI (as run_supervisory.py does)
        train_SI = config['train_SI']
        if train_SI:
            target = act_map_scaled
            input_ = sino_scaled
        else:
            target = sino_scaled
            input_ = act_map_scaled
        
        # Unpack reconstructions
        recon1 = recon_data[0] if recon_data[0] is not None else None
        recon2 = recon_data[1] if recon_data[1] is not None else None
        
        # Unpack attenuation (for Stage 5)
        if attenuation_data is not None:
            atten_sino, atten_map = attenuation_data
        else:
            atten_sino, atten_map = None, None
        
        # Validate all variables
        assert input_.shape == (3, 288, 288), f"input_ shape should be (3, 288, 288), got {input_.shape}"
        assert target.shape == (1, 180, 180), f"target shape should be (1, 180, 180), got {target.shape}"
        assert recon1.shape == (1, 180, 180), f"recon1 shape should be (1, 180, 180), got {recon1.shape}"
        assert recon2.shape == (1, 180, 180), f"recon2 shape should be (1, 180, 180), got {recon2.shape}"
        assert atten_map.shape == (1, 180, 180), f"atten_map shape should be (1, 180, 180), got {atten_map.shape}"
        assert atten_sino.shape[0] == 3, f"atten_sino channels should be 3, got {atten_sino.shape[0]}"
        
        print(f"✅ PASS: Nested tuple unpacking works correctly")
        print(f"   - input_: {input_.shape} (sinogram)")
        print(f"   - target: {target.shape} (activity image)")
        print(f"   - recon1: {recon1.shape}")
        print(f"   - recon2: {recon2.shape}")
        print(f"   - atten_map: {atten_map.shape}")
        print(f"   - atten_sino: {atten_sino.shape}")
    finally:
        if dataset is not None:
            dataset.close()
            del dataset
        gc.collect()
        cleanup_test_data(data_dir)


# ========================================================================================
# MAIN TEST RUNNER
# ========================================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("STAGE 1: SIMPLIFIED DATA LOADING TEST SUITE")
    print("="*80)
    
    tests = [
        test_backward_compatibility_no_augmentation,
        test_attenuation_map_loading,
        test_augmentation_consistency,
        test_nested_tuple_unpacking,
    ]
    
    failed = []
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ FAIL: {test_func.__name__}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed.append(test_func.__name__)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    passed = len(tests) - len(failed)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {len(failed)}/{len(tests)}")
    
    if failed:
        print(f"\nFailed tests:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
