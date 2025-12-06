from torch import nn
from ray import tune

#################################################################################################################################################################
## (config_RAY_SI OR config_RAY_IS) gets combined with (config_RAY_SUP or config_RAY_GAN) to form a single hyperparameter space for searching a single network ##
#################################################################################################################################################################

## Note: For the Coursera CycleGAN:
# gen_adv_criterion = disc_adv_criterion = nn.MSELoss()
# cycle_criterion = ident_criterion = nn.L1Loss()
# for notes on momentum, see: https://distill.pub/2017/momentum/

# --- Begin replacement for config_RAY_SI (flattened, dependent choices) ---
config_RAY_SI = { # Dictionary for Generator: Sinogram-->Image
    # Generator Network
    'SI_gen_mult': tune.uniform(1.1, 4),                        # Factor by which to multiply channels/block as one moves twowards the center of the network
    'SI_gen_fill': tune.choice([0,1,2]),                        # Number of constant-sized Conv2d layers/block
    'SI_gen_neck': tune.choice([1,6,11]),                       # Size of network neck: 1 = smallest, 11 = largest
    'SI_gen_z_dim': tune.lograndint(64, 4000),                  # If network utilizes smallest neck size (1x1 = a dense layer), this is the number of channels in the neck
    'SI_pad_mode': tune.choice(['zeros', 'reflect']),           # Padding type
    'SI_dropout': tune.choice([True,False]),                    # Implement dropout in network? (without cross-validation, this is likely never chosen)
    'SI_exp_kernel': tune.choice([3,4]),                        # Expanding kernel size: 3x3 or 4x4
    'SI_gen_hidden_dim': tune.lograndint(2, 30),                # Generator channel scaling factor. Larger numbers give more total channels.
    'SI_skip_mode': tune.choice(['none','add','concat']),       # Skip-connection mode

    # Discriminator Network
    'SI_disc_hidden_dim': tune.lograndint(10, 30),              # Discriminator channel scaling factor
    'SI_disc_patchGAN': tune.choice([True, False]),             # Use PatchGAN or not
    # Discriminator Optimizer
    'SI_disc_lr': tune.loguniform(1e-4,1e-2),
    'SI_disc_b1': tune.loguniform(0.1, 0.999),
    'SI_disc_b2': tune.loguniform(0.1, 0.999),
    'SI_disc_adv_criterion': tune.choice([nn.MSELoss(), nn.BCEWithLogitsLoss()]), # Possible options: tune.choice([nn.MSELoss(), nn.KLDivLoss(), nn.BCEWithLogitsLoss()]),
}

config_RAY_SI_learnScale = { # Dictionary for Generator: Sinogram-->Image with no normalization and learnable scaling
    ## Data Loading ##
    'SI_normalize': False,
    'SI_fixedScale': 1,                                          # Required by NPArrayDataLoader even when normalize=False. Set to 1 (no scaling).
    'SI_learnedScale_init': tune.loguniform(1e-6, 1e3),         # Initial guess for learned multiplier. Only used if SI_normalize=False. While SI_learnedScale_init is a hyperparamter, the actual scale is a learned parameter
    'SI_layer_norm': tune.choice(['none']),            # Could also add "group" normalization if you make it go into num_channels evenly.
    'SI_gen_final_activ': tune.choice([nn.ReLU(), None]),
}

config_RAY_SI_fixedScale = { # Dictionary for Generator: Sinogram-->Image with normalization and fixed scaling
    'SI_normalize': True,
    'SI_fixedScale': 1,
    'SI_layer_norm': tune.choice(['batch', 'instance', 'group', 'none']),
    'SI_gen_final_activ': tune.choice([nn.Tanh(), nn.Sigmoid(), nn.ReLU(), None]),
}

config_RAY_IS = { # Dictionary for Generator: Image-->Sinogram
    # Generator Network
    'IS_gen_mult': tune.uniform(1.1, 4),
    'IS_gen_fill': tune.choice([0,1,2]),
    'IS_gen_neck': tune.choice([1,6,11]),
    'IS_gen_z_dim': tune.lograndint(64, 4000),
    'IS_pad_mode': tune.choice(['zeros', 'reflect']),
    'IS_dropout': tune.choice([True,False]),
    'IS_exp_kernel': tune.choice([3,4]),
    'IS_gen_hidden_dim': tune.lograndint(2, 30),
    'IS_skip_mode': tune.choice(['none','add','concat']),

    # Discriminator Network
    'IS_disc_hidden_dim': tune.lograndint(10, 30),
    'IS_disc_patchGAN': tune.choice([True, False]),
    # Discriminator Optimizer
    'IS_disc_lr': tune.loguniform(1e-4,1e-2),
    'IS_disc_b1': tune.loguniform(0.1, 0.999),
    'IS_disc_b2': tune.loguniform(0.1, 0.999),
    'IS_disc_adv_criterion': tune.choice([nn.MSELoss(), nn.BCEWithLogitsLoss()]),
}

config_RAY_IS_learnScale = { # Dictionary for Generator: Sinogram-->Image with no normalization and learnable scaling
    ## Data Loading ##
    'IS_normalize': False,
    'IS_fixedScale': 1,        
    'IS_learnedScale_init': tune.loguniform(1e-6, 1e3),         # Initial guess for learned multiplier. Only used if IS_normalize=False. While IS_learnedScale_init is a hyperparamter, the actual scale is a learned parameter                                  # Required by NPArrayDataLoader even when normalize=False. Set to 1 (no scaling).
    'IS_layer_norm': tune.choice(['group', 'none']),
    'IS_gen_final_activ': tune.choice([nn.ReLU(), None]),
}

config_RAY_IS_fixedScale = { # Dictionary for Generator: Sinogram-->Image with normalization and fixed scaling
    'IS_normalize': True,
    'IS_fixedScale': 1,
    'IS_layer_norm': tune.choice(['batch', 'instance', 'group', 'none']),
    'IS_gen_final_activ': tune.choice([nn.Tanh(), nn.Sigmoid(), nn.ReLU(), None]),
}

config_RAY_SUP = { # This dictionary may be merged with either config_RAY_IS or config_RAY_SI to form a single dictionary for supervisory learning
    # NEW: New parameters added to config_RAY_SI (related to generator optimizer)
    'batch_base2_exponent': tune.randint(5, 10),  # Exponent for batch_size = 2^exponent (5->32, 6->64, 7->128, 8->256, 9->512)
    'gen_lr': tune.loguniform(1e-4,1e-2),
    'gen_b1': tune.loguniform(0.1, 0.999),
    'gen_b2': tune.loguniform(0.1, 0.999),
    'sup_criterion': tune.choice([nn.MSELoss(), nn.BCEWithLogitsLoss(), nn.L1Loss(), nn.KLDivLoss(reduction='batchmean')]), # Not SI or IS because used for both
    # OVERWRITES: overwrites values from config_RAY_SI or config_RAY_IS. This is done so time isn't wasted looking for unused hyperparameters.
    'SI_disc_hidden_dim': 1,
    'SI_disc_patchGAN': 1,
    'SI_disc_lr': 1,
    'SI_disc_b1': 1,
    'SI_disc_b2': 1,
    'SI_disc_adv_criterion': 1,
    'IS_disc_hidden_dim': 1,
    'IS_disc_patchGAN': 1,
    'IS_disc_lr': 1,
    'IS_disc_b1': 1,
    'IS_disc_b2': 1,
    'IS_disc_adv_criterion': 1,
    }

config_RAY_GAN = { # This is MERGED with either config_RAY_IS or config_RAY_SI to form a single dictionary for a generative adversarial network.
    # NEW
    'batch_base2_exponent': tune.randint(5, 10),  # Exponent for batch_size = 2^exponent (5->32, 6->64, 7->128, 8->256, 9->512)
    'gen_lr': tune.loguniform(1e-4,1e-2),
    'gen_b1': tune.loguniform(0.1, 0.999),
    'gen_b2': 0.999, #tune.loguniform(0.1, 0.999),
    'gen_adv_criterion': tune.choice([nn.MSELoss(), nn.BCEWithLogitsLoss()]),
    }

config_GAN_RAY_cycle = { # Mixed New/Overwrites (when combined with config_SI/config_IS) to form a single dictionary for a cycle-consistent generative adversarial network.
    # NEW
    'cycle_criterion': tune.choice([nn.MSELoss(), nn.L1Loss()]),
    'sup_criterion': tune.choice([nn.MSELoss(), nn.KLDivLoss(reduction='batchmean'), nn.L1Loss(), nn.BCEWithLogitsLoss()]),
    'lambda_adv': 1,
    'lambda_sup': 0,
    'lambda_cycle': 1,
    # OVERWRITES
    'gen_adv_criterion': nn.MSELoss(), #tune.choice([nn.MSELoss(), nn.KLDivLoss(), nn.BCEWithLogitsLoss()]),
    'IS_disc_lr': tune.loguniform(1e-4,1e-2),
    'SI_disc_lr': tune.loguniform(1e-4,1e-2),
    'batch_base2_exponent': tune.randint(5, 10),  # Exponent for batch_size = 2^exponent (5->32, 6->64, 7->128, 8->256, 9->512)
    'gen_lr': tune.loguniform(0.5e-4,1e-2),
    'gen_b1': tune.loguniform(0.1, 0.999),
    'gen_b2': 0.999, #tune.loguniform(0.1, 0.999),
    }

config_SUP_RAY_cycle = { # Mixed New/Overwrites (when combined with config_SI/config_IS) to form a single dictionary for a cycle-consistent, partially supervised network.
    # NEW
    'cycle_criterion': tune.choice([nn.MSELoss(), nn.L1Loss()]),
    'lambda_adv': 0,
    'lambda_sup': 1,
    'lambda_cycle':  tune.uniform(0, 10),
    # OVERWRITES
    'batch_base2_exponent': tune.randint(5, 10),  # Exponent for batch_size = 2^exponent (5->32, 6->64, 7->128, 8->256, 9->512)
    'gen_lr': tune.loguniform(0.5e-4,1e-2),
    'gen_b1': tune.loguniform(0.1, 0.999), # DCGan uses 0.5, https://distill.pub/2017/momentum/
    'gen_b2': tune.loguniform(0.1, 0.999),
    'sup_criterion': tune.choice([nn.MSELoss(), nn.KLDivLoss(), nn.L1Loss(), nn.BCEWithLogitsLoss()]),
    # NOT USED
    'gen_adv_criterion': nn.MSELoss(), #tune.choice([nn.MSELoss(), nn.KLDivLoss(), nn.BCEWithLogitsLoss()]),
    'IS_disc_lr': 1e-4, #tune.loguniform(1e-4,1e-2),
    'SI_disc_lr': 1e-4, #tune.loguniform(1e-4,1e-2),
    }