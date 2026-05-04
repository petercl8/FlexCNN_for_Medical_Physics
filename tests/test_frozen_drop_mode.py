"""
Unit tests for frozen feature dropout modes (per_channel vs all_or_none).

Tests verify that:
1. per_channel mode drops each channel independently.
2. all_or_none mode drops all or no channels as a batch.
3. test/visualize modes always return features unchanged.
"""
import torch
import pytest
from FlexCNN_for_Medical_Physics.functions.main_run_functions.train_utils import optionally_drop_some_channels


def create_test_features(batch_size=2, channels=4, spatial_size=8, device='cpu'):
    """Create dummy encoder/decoder feature tuples for testing."""
    feat1 = torch.ones(batch_size, channels, spatial_size, spatial_size, device=device)
    feat2 = torch.ones(batch_size, channels, spatial_size, spatial_size, device=device)
    return (feat1, feat2), (feat1.clone(), feat2.clone())


class TestPerChannelDropout:
    """Test per-channel dropout mode."""
    
    def test_per_channel_p_drop_zero(self):
        """With p_drop=0, no channels should be dropped."""
        enc_feats, dec_feats = create_test_features()
        injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
            enc_feats, dec_feats, run_mode='train', p_drop=0.0, frozen_drop_mode='per_channel'
        )
        
        # All features should be unchanged
        assert torch.allclose(injected_enc[0], enc_feats[0])
        assert torch.allclose(injected_dec[0], dec_feats[0])
        assert dropped_percent == 0.0
    
    def test_per_channel_p_drop_one(self):
        """With p_drop=1.0, all channels should be dropped."""
        enc_feats, dec_feats = create_test_features()
        injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
            enc_feats, dec_feats, run_mode='train', p_drop=1.0, frozen_drop_mode='per_channel'
        )
        
        # All features should be zeros
        assert torch.allclose(injected_enc[0], torch.zeros_like(injected_enc[0]))
        assert torch.allclose(injected_dec[0], torch.zeros_like(injected_dec[0]))
        assert dropped_percent == 100.0
    
    def test_per_channel_intermediate_p_drop(self):
        """With intermediate p_drop (e.g., 0.5), roughly half channels should be dropped (stochastic)."""
        enc_feats, dec_feats = create_test_features(channels=100)  # More channels for statistics
        
        # Run multiple times and check average dropout rate
        dropout_rates = []
        for _ in range(10):
            injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
                enc_feats, dec_feats, run_mode='train', p_drop=0.5, frozen_drop_mode='per_channel'
            )
            dropout_rates.append(dropped_percent)
        
        # Average should be close to 50%
        avg_dropout = sum(dropout_rates) / len(dropout_rates)
        assert 30.0 < avg_dropout < 70.0, f"Expected ~50%, got {avg_dropout}%"


class TestAllOrNoneDropout:
    """Test all-or-none dropout mode."""
    
    def test_all_or_none_p_drop_zero(self):
        """With p_drop=0, no channels should be dropped."""
        enc_feats, dec_feats = create_test_features()
        injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
            enc_feats, dec_feats, run_mode='train', p_drop=0.0, frozen_drop_mode='all_or_none'
        )
        
        # All features should be unchanged
        assert torch.allclose(injected_enc[0], enc_feats[0])
        assert torch.allclose(injected_dec[0], dec_feats[0])
        assert dropped_percent == 0.0
    
    def test_all_or_none_p_drop_one(self):
        """With p_drop=1.0, all channels should be dropped."""
        enc_feats, dec_feats = create_test_features()
        injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
            enc_feats, dec_feats, run_mode='train', p_drop=1.0, frozen_drop_mode='all_or_none'
        )
        
        # All features should be zeros
        assert torch.allclose(injected_enc[0], torch.zeros_like(injected_enc[0]))
        assert torch.allclose(injected_dec[0], torch.zeros_like(injected_dec[0]))
        assert dropped_percent == 100.0
    
    def test_all_or_none_binary_decision(self):
        """All-or-none should produce either 0% or 100% dropout, never in between."""
        enc_feats, dec_feats = create_test_features(channels=50)
        
        # Run multiple times and check that dropout_percent is always 0 or 100
        for _ in range(20):
            injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
                enc_feats, dec_feats, run_mode='train', p_drop=0.5, frozen_drop_mode='all_or_none'
            )
            
            assert dropped_percent in [0.0, 100.0], f"Expected 0% or 100%, got {dropped_percent}%"


class TestTestAndVisualizeModeBehavior:
    """Test that test/visualize modes always return features unchanged."""
    
    def test_test_mode_no_dropout(self):
        """Test mode should return features unchanged regardless of p_drop."""
        enc_feats, dec_feats = create_test_features()
        enc_feats_orig = (enc_feats[0].clone(), enc_feats[1].clone())
        dec_feats_orig = (dec_feats[0].clone(), dec_feats[1].clone())
        
        for p_drop in [0.0, 0.5, 1.0]:
            for mode in ['per_channel', 'all_or_none']:
                injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
                    enc_feats, dec_feats, run_mode='test', p_drop=p_drop, frozen_drop_mode=mode
                )
                
                assert torch.allclose(injected_enc[0], enc_feats_orig[0])
                assert torch.allclose(injected_dec[0], dec_feats_orig[0])
                assert dropped_percent == 0.0, f"Expected 0% dropout in test mode, got {dropped_percent}%"
    
    def test_visualize_mode_no_dropout(self):
        """Visualize mode should return features unchanged regardless of p_drop."""
        enc_feats, dec_feats = create_test_features()
        enc_feats_orig = (enc_feats[0].clone(), enc_feats[1].clone())
        dec_feats_orig = (dec_feats[0].clone(), dec_feats[1].clone())
        
        for p_drop in [0.0, 0.5, 1.0]:
            for mode in ['per_channel', 'all_or_none']:
                injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
                    enc_feats, dec_feats, run_mode='visualize', p_drop=p_drop, frozen_drop_mode=mode
                )
                
                assert torch.allclose(injected_enc[0], enc_feats_orig[0])
                assert torch.allclose(injected_dec[0], dec_feats_orig[0])
                assert dropped_percent == 0.0, f"Expected 0% dropout in visualize mode, got {dropped_percent}%"


class TestNoneFeatures:
    """Test handling of None features."""
    
    def test_none_encoder_features_per_channel(self):
        """Per-channel mode should handle None encoder features."""
        enc_feats, dec_feats = create_test_features()
        
        injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
            None, dec_feats, run_mode='train', p_drop=0.5, frozen_drop_mode='per_channel'
        )
        
        assert injected_enc is None
        assert injected_dec is not None
    
    def test_none_features_all_or_none(self):
        """All-or-none mode should handle None features."""
        enc_feats, dec_feats = create_test_features()
        
        injected_enc, injected_dec, dropped_percent = optionally_drop_some_channels(
            None, dec_feats, run_mode='train', p_drop=0.5, frozen_drop_mode='all_or_none'
        )
        
        assert injected_enc is None
        assert injected_dec is not None


class TestInvalidMode:
    """Test error handling for invalid modes."""
    
    def test_invalid_frozen_drop_mode(self):
        """Invalid frozen_drop_mode should raise ValueError."""
        enc_feats, dec_feats = create_test_features()
        
        with pytest.raises(ValueError, match="Invalid frozen_drop_mode"):
            optionally_drop_some_channels(
                enc_feats, dec_feats, run_mode='train', p_drop=0.5, frozen_drop_mode='invalid_mode'
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
