from torch import nn

config_CYCLEGAN={ # Works, yeah! ("4a92")
    "network_type": "CYCLEGAN",
    "SI_stats_criterion": None,
    "SI_alpha_min": -1,
    "SI_half_life_examples": 2000,
    "IS_stats_criterion": None,
    "IS_alpha_min": -1,
    "IS_half_life_examples": 2000,
    "IS_disc_adv_criterion": nn.BCEWithLogitsLoss(),
    "IS_disc_b1": 0.3335905891003811,
    "IS_disc_b2": 0.999,
    "IS_disc_hidden_dim": 11,
    "IS_disc_lr": 0.0006554051278271163,
    "IS_disc_patchGAN": True,
    "IS_dropout": False,
    "IS_exp_kernel": 3,
    "IS_gen_fill": 0,
    "IS_gen_final_activ": None,
    "IS_gen_hidden_dim": 15,
    "IS_gen_mult": 3,
    "IS_gen_neck": 11,
    "IS_gen_z_dim": 5,
    "IS_layer_norm": "instance",
    "IS_normalize": False,
    "IS_pad_mode": "reflect",
    "SI_fixedScale": 1,
    "SI_disc_adv_criterion": nn.MSELoss(),
    "SI_disc_b1": 0.102081,
    "SI_disc_b2": 0.999,
    "SI_disc_hidden_dim": 25,
    "SI_disc_lr": 0.0005793968896471209,
    "SI_disc_patchGAN": True,
    "SI_dropout": False,
    "SI_exp_kernel": 3,
    "SI_gen_fill": 0,
    "SI_gen_final_activ": nn.Sigmoid(),
    "SI_gen_hidden_dim": 46,
    "SI_gen_mult": 1.41,
    "SI_gen_neck": 1,
    "SI_gen_z_dim": 115,
    "SI_layer_norm": "batch",
    "SI_normalize": True,
    "SI_pad_mode": "reflect",
    "SI_fixedScale": 1400,
    "batch_size": 91,
    "cycle_criterion": nn.MSELoss(),
    "gen_adv_criterion": nn.MSELoss(),
    "gen_b1": 0.1610671788990834,
    "gen_b2": 0.999,
    "gen_lr": 0.0023450700434171526,
    "lambda_adv": 1,
    "lambda_cycle": 1, #1
    "lambda_sup": 0, # 0
    "sup_base_criterion": nn.L1Loss()
    }

'''
## Was best for training the CycleGAN all at once ##
# Below config is "SM_1662", the lowest optim_metric in 9h run, 90x90 symmetrical (not symmetrized parameters) networks.
# It was trained on IO_channels==3 but seems to work fine for IO_channels==1. Also, both discriminators use the same architecture,
# which is really better suited for the sinogram (Disc_S_90).
#
# Lessons Learned:
# 1) Utilized: different size necks, final activations, channels, patchGAN
# 2) NOT Uilized: fill Conv2d layers, different adv_criterion (for disc loss), different normalizations


config={ # Symmetrize == FALSE (final activations don't match). This was the best over full tune train time (9 hours). Use two Sinogram discriminators.
    'batch_size': 107,
    'gen_b1': 0.339,
    'gen_b2': 0.999,
    'gen_lr': 0.000103,
    "cycle_criterion": nn.L1Loss(),
    "sup_base_criterion": nn.L1Loss(),
    "gen_adv_criterion": nn.KLDivLoss(),
    "lambda_adv": 1,
    "lambda_cycle": 2,
    "lambda_sup": 0,

    "IS_disc_adv_criterion": nn.MSELoss(),
    "IS_disc_b1": 0.19520417398460468,
    "IS_disc_b2": 0.999,
    "IS_disc_hidden_dim": 23,
    "IS_disc_lr": 0.0022230036964765274,  # disc_lr is 10X faster than SI
    "IS_disc_patchGAN": False,            # true for SI (make sense, images can be more true/false in patches)
    "IS_gen_fill": 0,                     # fill=0 for both IS and SI
    "IS_gen_final_activ": nn.Sigmoid(),   # tuned final activations opposite than for GANs
    "IS_gen_hidden_dim": 8,               # IS much less complex than SI (8 vs 16 hidden_dim)
    "IS_gen_mult": 3,                     # mult=3 for both IS and SI
    "IS_gen_z_dim": 5,
    "IS_normalize": True,                 # both are normalized here
    "SI_fixedScale": 1,                        # OMG, this is weird. We are normalizing both, but the SI image scale is 1400x the IS. Could this by why final activation is now Sigmoid?

    'IS_layer_norm': 'batch', # Batch
    'IS_pad_mode': 'reflect',
    'IS_dropout': False,
    "IS_gen_neck": 5,            # 2
    'IS_exp_kernel': 4,          # 4

    "SI_disc_adv_criterion": nn.MSELoss(),
    "SI_disc_b1": 0.30423542819878224,
    "SI_disc_b2": 0.999,
    "SI_disc_hidden_dim": 23,
    "SI_disc_lr": 0.00020737432489437965,
    "SI_disc_patchGAN": True,
    "SI_gen_fill": 0,
    "SI_gen_final_activ": nn.Tanh(),
    "SI_gen_hidden_dim": 22,
    "SI_gen_mult": 3,
    "SI_gen_z_dim": 1195,                 # Represents an 8x drop in information into narrowest part of neck
    "SI_normalize": True, # True
    "SI_fixedScale": 1400,

    'SI_layer_norm': 'batch',
    'SI_pad_mode': 'reflect',
    'SI_dropout': False,
    "SI_gen_neck": 1,            # 1
    'SI_exp_kernel': 4,          # 4
    }
'''