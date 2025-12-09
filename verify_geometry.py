"""
Verify Generator geometry: skip connection dimensions and final output size.
Tests all 18 combinations of SI_exp_kernel, SI_skip_mode, and SI_gen_neck.
"""

import torch
import sys
sys.path.insert(0, '/content/FlexCNN_for_Medical_Physics')

from FlexCNN_for_Medical_Physics.classes.generators import Generator

# Test configuration
base_config = {
    'sino_size': 180,
    'image_size': 180,
    'sino_channels': 1,
    'image_channels': 1,
    'SI_gen_hidden_dim': 4,  # Small for fast testing
    'SI_gen_mult': 1.0,
    'SI_gen_z_dim': 16,
    'SI_dropout': False,
    'SI_layer_norm': 'none',
    'SI_pad_mode': 'zeros',
    'SI_normalize': False,
    'SI_fixedScale': 1.0,
    'SI_gen_final_activ': None,
    'train_SI': True,
}

exp_kernels = [3, 4]
skip_modes = ['none', 'add', 'concat']
neck_values = [1, 6, 11]

print("=" * 100)
print("GENERATOR GEOMETRY VERIFICATION")
print("=" * 100)

all_pass = True

for exp_kernel in exp_kernels:
    for skip_mode in skip_modes:
        for neck in [1, 6, 11]:
            config = base_config.copy()
            config['SI_exp_kernel'] = exp_kernel
            config['SI_skip_mode'] = skip_mode
            config['SI_gen_neck'] = neck
            
            try:
                gen = Generator(config=config, gen_SI=True)
                
                # Test forward pass with dummy input
                dummy_input = torch.randn(2, 1, 180, 180)
                output = gen(dummy_input)
                
                output_size = output.shape[-1]
                if output_size == 180:
                    status = "✓ PASS"
                else:
                    status = f"✗ FAIL (output {output_size}×{output_size})"
                    all_pass = False
                
                print(f"exp_kernel={exp_kernel}, skip_mode={skip_mode:6s}, neck={neck:2d}: {status}")
                
            except Exception as e:
                print(f"exp_kernel={exp_kernel}, skip_mode={skip_mode:6s}, neck={neck:2d}: ✗ ERROR - {str(e)[:60]}")
                all_pass = False

print("=" * 100)
if all_pass:
    print("All combinations passed!")
else:
    print("Some combinations failed. See details above.")
print("=" * 100)
