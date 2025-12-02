from torch import nn

config_GAN_SI = { # Older, this still outperforms the more recent tuning
    'SI_disc_adv_criterion': nn.MSELoss(),
    'SI_normalize': True, # True
    'SI_fixedScale': 1400, # 1      # Added later
    'SI_gen_neck': 1, # 1
    'SI_layer_norm': 'batch',
    'SI_pad_mode': 'reflect',
    'SI_dropout': False,
    'SI_exp_kernel': 3,
    'SI_gen_fill': 0,
    'SI_gen_mult': 1.41,
    'SI_gen_z_dim': 115,
    'SI_gen_final_activ': nn.Sigmoid(),
    'SI_disc_patchGAN': True,
    'SI_gen_hidden_dim': 46,
    'SI_disc_hidden_dim': 25,
    'SI_disc_b1': 0.102081,
    'SI_disc_b2': 0.999,
    # Gets Overwritten Below
    'SI_disc_lr': 0.000167384,
    'batch_size': 78,
    'gen_adv_criterion': nn.MSELoss(),
    'gen_lr': 0.000167384,
    'gen_b1': 0.102081,
    'gen_b2': 0.999,
    }

config_GAN_IS = { # new, looks good by step 400, somewhat blocky. May be outperforming config_GAN_SI
  "IS_disc_adv_criterion": nn.BCEWithLogitsLoss(),
  "IS_disc_b1": 0.3335905891003811,
  "IS_disc_b2": 0.999,
  "IS_disc_hidden_dim": 11,
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
  # Gets Overwritten Below
  "IS_disc_lr": 0.00021705437338035208,
  "batch_size": 88,
  "gen_adv_criterion": nn.MSELoss(),
  "gen_b1": 0.46293297275979556,
  "gen_b2": 0.999,
  "gen_lr": 0.00042810775483742824
}

'''
# this config looks decent at step 1100, worse at 1440, better at 1900, etc. (variable). It isn't blocky.
config_GAN_IS_old = {
    "batch_size": 82,
    "gen_adv_criterion": nn.BCEWithLogitsLoss(),
    "gen_lr": 3.365297856241193e-05,
    "gen_b1": 0.11790916451301556,
    "gen_b2": 0.999,

    "IS_disc_adv_criterion": nn.BCEWithLogitsLoss(),
    "IS_normalize": False, # FALSE
    'SI_fixedScale': 1, # 1
    'IS_gen_mult': 3,
    'IS_gen_fill': 0,
    'IS_gen_neck': 5, # Wide neck
    "IS_gen_z_dim": 115,
    'IS_layer_norm': 'instance',
    'IS_pad_mode': 'reflect',
    'IS_dropout': False,
    'IS_exp_kernel': 4,
    "IS_gen_final_activ": nn.Tanh(), # nn.Tanh()
    "IS_disc_patchGAN": True,
    "IS_gen_hidden_dim": 16,
    "IS_disc_hidden_dim": 19,
    "IS_disc_lr": 0.00020392229473545828,
    "IS_disc_b1": 0.35984156365558084,
    "IS_disc_b2": 0.999,
    }
'''