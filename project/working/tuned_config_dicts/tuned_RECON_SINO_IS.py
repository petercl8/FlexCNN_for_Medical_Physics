# Maps 3D OSEM reconstructions to sinograms.
# 320x320 Network.
# Data: 288x257, SINOGRAM padded to 320x320, tuned SSIM
#       Crop sinograms vertically to 288, then bilinearly resize horizontally to size 257.
#       results in 288x257 size which is then padded sinoram-style horizontally to 320
# Network then maps 3D OSEM reconstructions to 3D OSEM sinograms
config_RECON_SINO_IS = {
  "IS_alpha_min": -1,
  "IS_dropout": False,
  "IS_exp_kernel": 4,
  "IS_fixedScale": 1,
  "IS_gen_fill": 0,
  "IS_gen_final_activ": "LeakyReLU",
  "IS_gen_hidden_dim": 10,
  "IS_gen_mult": 2.1398717769491116,
  "IS_gen_neck": "narrow",
  "IS_gen_z_dim": 657,
  "IS_half_life_examples": -1,
  "IS_layer_norm": "none",
  "IS_learnedScale_init": 10.310436748113611,
  "IS_moment_1_fraction": -1,
  "IS_normalize": False,
  "IS_output_scale_lr_mult": 1.4951972849973978,
  "IS_pad_mode": "replicate",
  "IS_skip_mode": "none",
  "IS_stats_criterion": -1,
  "SI_disc_adv_criterion": 1,
  "SI_disc_b1": 1,
  "SI_disc_b2": 1,
  "SI_disc_hidden_dim": 1,
  "SI_disc_lr": 1,
  "SI_disc_patchGAN": 1,
  "batch_base2_exponent": 5,
  "frozen_variant": "RECON_SINO",
  "gen_b1": 0.2037541246627113,
  "gen_b2": 0.17111422257861728,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.00026443268866718967,
  "gen_sino_channels_IS": 1,
  "gen_sino_channels_SI": 3,
  "gen_sino_size": 320,
  "network_type": "RECON_SINO",
  "recon_variant": 2,
  "sup_base_criterion": "MSELoss",
  "train_SI": False
}

'''
# Maps FORE reconstructions to sinograms.
# 320x320 Network.
# Data: 288x257, SINOGRAM padded to 320x320, tuned SSIM
#       Crop sinograms vertically to 288, then bilinearly resize horizontally to size 257.
#       results in 288x257 size which is then padded sinoram-style horizontally to 320
# Network then maps FORE reconstructions to FORE sinograms
config_RECON_SINO_IS = {
  "IS_alpha_min": -1,
  "IS_dropout": False,
  "IS_exp_kernel": 4,
  "IS_fixedScale": 1,
  "IS_gen_fill": 0,
  "IS_gen_final_activ": None,
  "IS_gen_hidden_dim": 30,
  "IS_gen_mult": 2.516246561597289,
  "IS_gen_neck": "medium",
  "IS_gen_z_dim": 1764,
  "IS_half_life_examples": -1,
  "IS_layer_norm": "none",
  "IS_learnedScale_init": 6.042211851495784,
  "IS_moment_1_fraction": -1,
  "IS_normalize": False,
  "IS_output_scale_lr_mult": 4.7377521066712776,
  "IS_pad_mode": "zeros",
  "IS_skip_mode": "none",
  "IS_stats_criterion": -1,
  "SI_disc_adv_criterion": 1,
  "SI_disc_b1": 1,
  "SI_disc_b2": 1,
  "SI_disc_hidden_dim": 1,
  "SI_disc_lr": 1,
  "SI_disc_patchGAN": 1,
  "batch_base2_exponent": 6,
  "frozen_variant": "RECON_SINO",
  "gen_b1": 0.6826187985646257,
  "gen_b2": 0.8087876386292491,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.0007764276587482484,
  "gen_sino_channels_IS": 1,
  "gen_sino_channels_SI": 3,
  "gen_sino_size": 320,
  "network_type": "RECON_SINO",
  "recon_variant": 1,
  "sup_base_criterion": "MSELoss",
  "train_SI": False
}
'''