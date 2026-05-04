"""
Simple validation script for frozen dropout modes (no pytest required).
"""
import torch
import sys

# Add the project to path
sys.path.insert(0, r'C:\Users\Peter Lindstrom\Desktop\FlexCNN_for_Medical_Physics')

from FlexCNN_for_Medical_Physics.functions.main_run_functions.train_utils import optionally_drop_some_channels


def create_test_features(batch_size=2, channels=4, spatial_size=8, device='cpu'):
    """Create dummy encoder/decoder feature tuples for testing."""
    feat1 = torch.ones(batch_size, channels, spatial_size, spatial_size, device=device)
    feat2 = torch.ones(batch_size, channels, spatial_size, spatial_size, device=device)
    return (feat1, feat2), (feat1.clone(), feat2.clone())


print("=" * 80)
print("FROZEN DROPOUT MODE VALIDATION TESTS")
print("=" * 80)

# Test 1: Per-channel with p_drop=0
print("\n[TEST 1] Per-channel mode with p_drop=0 (no dropout)")
enc_feats, dec_feats = create_test_features()
injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
    enc_feats, dec_feats, run_mode='train', p_drop=0.0, frozen_drop_mode='per_channel'
)
assert torch.allclose(injected_enc[0], enc_feats[0]), "Features should be unchanged"
assert dropped_percent == 0.0, "Dropped percent should be 0"
print("✓ PASS: Features unchanged, dropout = 0%")

# Test 2: Per-channel with p_drop=1.0
print("\n[TEST 2] Per-channel mode with p_drop=1.0 (all dropout)")
enc_feats, dec_feats = create_test_features()
injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
    enc_feats, dec_feats, run_mode='train', p_drop=1.0, frozen_drop_mode='per_channel'
)
assert torch.allclose(injected_enc[0], torch.zeros_like(injected_enc[0])), "All features should be zero"
assert dropped_percent == 100.0, "Dropped percent should be 100"
print("✓ PASS: All features zeroed, dropout = 100%")

# Test 3: All-or-none with p_drop=0
print("\n[TEST 3] All-or-none mode with p_drop=0 (no dropout)")
enc_feats, dec_feats = create_test_features()
injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
    enc_feats, dec_feats, run_mode='train', p_drop=0.0, frozen_drop_mode='all_or_none'
)
assert torch.allclose(injected_enc[0], enc_feats[0]), "Features should be unchanged"
assert dropped_percent == 0.0, "Dropped percent should be 0"
print("✓ PASS: Features unchanged, dropout = 0%")

# Test 4: All-or-none with p_drop=1.0
print("\n[TEST 4] All-or-none mode with p_drop=1.0 (all dropout)")
enc_feats, dec_feats = create_test_features()
injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
    enc_feats, dec_feats, run_mode='train', p_drop=1.0, frozen_drop_mode='all_or_none'
)
assert torch.allclose(injected_enc[0], torch.zeros_like(injected_enc[0])), "All features should be zero"
assert dropped_percent == 100.0, "Dropped percent should be 100"
print("✓ PASS: All features zeroed, dropout = 100%")

# Test 5: All-or-none produces binary outcomes (0 or 100)
print("\n[TEST 5] All-or-none mode binary decision (0% or 100%)")
enc_feats, dec_feats = create_test_features(channels=50)
outcomes = []
for trial in range(20):
    injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
        enc_feats, dec_feats, run_mode='train', p_drop=0.5, frozen_drop_mode='all_or_none'
    )
    outcomes.append(dropped_percent)
    assert dropped_percent in [0.0, 100.0], f"Expected 0% or 100%, got {dropped_percent}%"

has_zero = any(x == 0.0 for x in outcomes)
has_hundred = any(x == 100.0 for x in outcomes)
print(f"✓ PASS: {len([x for x in outcomes if x==0.0])} trials with 0%, {len([x for x in outcomes if x==100.0])} trials with 100%")

# Test 6: Test mode always returns unchanged
print("\n[TEST 6] Test mode always returns features unchanged (p_drop=1.0)")
enc_feats, dec_feats = create_test_features()
enc_feats_orig = (enc_feats[0].clone(), enc_feats[1].clone())
dec_feats_orig = (dec_feats[0].clone(), dec_feats[1].clone())

for mode in ['per_channel', 'all_or_none']:
    injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
        enc_feats, dec_feats, run_mode='test', p_drop=1.0, frozen_drop_mode=mode
    )
    assert torch.allclose(injected_enc[0], enc_feats_orig[0]), "Features should be unchanged in test mode"
    assert dropped_percent == 0.0, "Dropped percent should be 0 in test mode"

print("✓ PASS: Both modes return unchanged features in test mode")

# Test 7: Visualize mode always returns unchanged
print("\n[TEST 7] Visualize mode always returns features unchanged (p_drop=1.0)")
enc_feats, dec_feats = create_test_features()
enc_feats_orig = (enc_feats[0].clone(), enc_feats[1].clone())
dec_feats_orig = (dec_feats[0].clone(), dec_feats[1].clone())

for mode in ['per_channel', 'all_or_none']:
    injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
        enc_feats, dec_feats, run_mode='visualize', p_drop=1.0, frozen_drop_mode=mode
    )
    assert torch.allclose(injected_enc[0], enc_feats_orig[0]), "Features should be unchanged in visualize mode"
    assert dropped_percent == 0.0, "Dropped percent should be 0 in visualize mode"

print("✓ PASS: Both modes return unchanged features in visualize mode")

# Test 8: Invalid mode raises error
print("\n[TEST 8] Invalid frozen_drop_mode raises ValueError")
enc_feats, dec_feats = create_test_features()
try:
    optionally_drop_some_channels(
        enc_feats, dec_feats, run_mode='train', p_drop=0.5, frozen_drop_mode='invalid_mode'
    )
    print("✗ FAIL: Should have raised ValueError")
    sys.exit(1)
except ValueError as e:
    print(f"✓ PASS: Raised ValueError as expected: {str(e)[:60]}...")

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
