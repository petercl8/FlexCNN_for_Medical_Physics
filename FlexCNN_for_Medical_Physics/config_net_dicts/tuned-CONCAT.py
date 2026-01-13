config_CONCAT={
    # CONCAT mode: Activity (3ch) + Attenuation (1ch) concatenated -> 4ch sinogram input
    # User must specify sino_channels=4 in network_opts to match concatenated input
    # Architecture tuned for sinogram->image reconstruction with concatenated inputs
    "SI_dropout": False,
    "SI_exp_kernel": 3,
    "SI_fixedScale": 1,
    "SI_gen_fill": 1,
    "SI_gen_final_activ": "ELU",
    "SI_gen_hidden_dim": 32,
    "SI_gen_mult": 1.5,
    "SI_gen_neck": 12,
    "SI_gen_z_dim": 512,
    "SI_layer_norm": "instance",
    "SI_learnedScale_init": 20.0,
    "SI_normalize": False,
    "SI_pad_mode": "zeros",
    "SI_skip_mode": "none",
    "batch_base2_exponent": 6,
    "gen_b1": 0.35,
    "gen_b2": 0.11,
    "gen_lr": 0.0006,
    "sup_criterion": "MSELoss",
    "train_SI": True  # CONCAT always trains sinogram->image direction
} 