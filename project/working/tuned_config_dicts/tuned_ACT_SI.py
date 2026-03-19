from sympy import false, true
from torch import nn

## Supervisory
'''
In this module, set the correct hyperparameter dictionary for config_SUP_SI.
This is the dictionary of hyperparameters that determines the form of a the network that will be trained, tested, or visualized (when doing supervised learning, Sinogram-->Image).
You will usually find these hyperparameters by performing tuning and examining the best performing networks in tensorboard.

If training supervisory loss networks only, you don't need to worry about the other dictionaries in this section (GANs, Cycle-Consistent).
You also don't need to worry about "Search Spaces", as this is simply a dictionary of the search space that Ray Tune uses when tuning.
Feel free to look at it though, to see how I set up the search space. The last section (Set Correct Config) is where the configuration dictionary gets assigned.
The dictionary is either a searchable space, if tuning, or a set of fixed hyperparameters, if training, testing, or visualizing the data set.
'''


# (I) 320x320 Network. Data: 288x257, SINOGRAM padded to 288x288, tuned SSIM
# Crop sinograms vertically to 288, then bilinearly resize horizontally to size 257.
# results in 288x257 size which is then padded sinoram-style horizontally to 288
config_ACT_SI = {
  "SI_alpha_min": -1,
  "SI_disc_adv_criterion": 1,
  "SI_dropout": False,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": "LeakyReLU",
  "SI_gen_hidden_dim": 27,
  "SI_gen_mult": 2.346554155198677,
  "SI_gen_neck": "narrow",
  "SI_gen_z_dim": 1275,
  "SI_half_life_examples": -1,
  "SI_layer_norm": "none",
  "SI_learnedScale_init": 22.311191399467425,
  "SI_moment_1_fraction": -1,
  "SI_normalize": False,
  "SI_output_scale_lr_mult": 1.1652951981214295,
  "SI_pad_mode": "replicate",
  "SI_skip_mode": "none",
  "SI_stats_criterion": -1,
  "batch_base2_exponent": 5,
  "frozen_variant": "RECON_SINO",
  "gen_b1": 0.16376752129041514,
  "gen_b2": 0.6314511499737464,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.00021715942590932197,
  "gen_sino_channels": 3,
  "gen_sino_size": 320,
  "network_type": "ACT",
  "recon_variant": 1,
  "sup_base_criterion": "MSELoss",
  "train_SI": True
}

#####################################
### 288x288 Network Tunings Below ###
#####################################

'''
# (H) 288x257, SINOGRAM padded to 288x288, tuned SSIM, DROPOUT=True enforced 
# Crop sinograms vertically to 288, then bilinearly resize horizontally to size 257.
# results in 288x257 size which is then padded sinoram-style horizontally to 288
config_ACT_SI = { 
  "SI_alpha_min": -1,
  "SI_dropout": True,
  "SI_exp_kernel": 4,
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": None,
  "SI_gen_hidden_dim": 17,
  "SI_gen_mult": 3.235794135077702,
  "SI_gen_neck": "medium",
  "SI_gen_z_dim": 776,
  "SI_half_life_examples": -1,
  "SI_layer_norm": "instance",
  "SI_learnedScale_init": 0.6598989774160382,
  "SI_moment_1_fraction": -1,
  "SI_normalize": False,
  "SI_output_scale_lr_mult": 1.1381973993890924,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "conv",
  "SI_stats_criterion": -1,
  "batch_base2_exponent": 5,
  "frozen_variant": "RECON_SINO",
  "gen_b1": 0.6315686454122481,
  "gen_b2": 0.4796586315253887,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.004073671882171635,
  "gen_sino_channels": 3,
  "gen_sino_size": 288,
  "network_type": "ACT",
  "recon_variant": 1,
  "sup_base_criterion": "MSELoss",
  "train_SI": True
}
'''

'''
# (G) 288x180, SINOGRAM padded to 288x288, tuned SSIM 
# Crop sinograms vertically to 288, then bilinearly resize horizontally to size 180.
# results in 288x180 size which is then sinogram padded horizontally to 288
config_ACT_SI = { 
  "SI_alpha_min": -1,
  "SI_dropout": False,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": None,
  "SI_gen_hidden_dim": 16,
  "SI_gen_mult": 2.3555295003025276,
  "SI_gen_neck": "medium",
  "SI_gen_z_dim": 576,
  "SI_half_life_examples": -1,
  "SI_layer_norm": "instance",
  "SI_learnedScale_init": 10.893989027618026,
  "SI_moment_1_fraction": -1,
  "SI_normalize": False,
  "SI_output_scale_lr_mult": 1.1618819638475886,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "conv",
  "SI_stats_criterion": -1,
  "batch_base2_exponent": 5,
  "gen_b1": 0.8174588820756479,
  "gen_b2": 0.7294818348374883,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.0001232534342696305,
  "gen_sino_channels": 3,
  "gen_sino_size": 288,
  "network_type": "ACT",
  "sup_base_criterion": "MSELoss",
  "train_SI": True
}
'''

'''
# (F) 288x218, SINOGRAM padded to 288x288, tuned SSIM 
# Crop sinograms vertically to 288, then bilinearly resize horizontally to size 218.
# results in 288x218 size which is then sinogram padded horizontally to 288
config_ACT_SI = { 
  "SI_alpha_min": -1,
  "SI_dropout": False,
  "SI_exp_kernel": 4,
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": "LeakyReLU",
  "SI_gen_hidden_dim": 34,
  "SI_gen_mult": 1.6784499724995354,
  "SI_gen_neck": "wide",
  "SI_gen_z_dim": 1733,
  "SI_half_life_examples": -1,
  "SI_layer_norm": "instance",
  "SI_learnedScale_init": 3.1060983465329053,
  "SI_moment_1_fraction": -1,
  "SI_normalize": False,
  "SI_output_scale_lr_mult": 4.134778822027846,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "none",
  "SI_stats_criterion": -1,
  "batch_base2_exponent": 5,
  "gen_b1": 0.6099253222709485,
  "gen_b2": 0.2194047349693382,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.0003521818874320586,
  "gen_sino_channels": 3,
  "gen_sino_size": 288,
  "network_type": "ACT",
  "sup_base_criterion": "MSELoss",
  "train_SI": True
}
'''
'''
# (E) 288x257, SINOGRAM padded to 288x288, tuned SSIM 
# Crop sinograms vertically to 288, then bilinearly resize horizontally to size 257.
# results in 288x257 size which is then padded sinoram-style horizontally to 288
config_ACT_SI = { 
  "SI_alpha_min": -1,
  "SI_dropout": False,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": "LeakyReLU",
  "SI_gen_hidden_dim": 20,
  "SI_gen_mult": 2.1572554323300173,
  "SI_gen_neck": "wide",
  "SI_gen_z_dim": 1939,
  "SI_half_life_examples": -1,
  "SI_layer_norm": "instance",
  "SI_learnedScale_init": 4.981741020141211,
  "SI_moment_1_fraction": -1,
  "SI_normalize": False,
  "SI_output_scale_lr_mult": 4.290911545096751,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "conv",
  "SI_stats_criterion": -1,
  "batch_base2_exponent": 5,
  "gen_b1": 0.19220905219147658,
  "gen_b2": 0.1767702014809058,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.000330514188658911,
  "gen_sino_channels": 3,
  "gen_sino_size": 288,  #288
  "network_type": "ACT",
  "sup_base_criterion": "MSELoss",
  "train_SI": True
}
'''
'''
# (D, B) 288x257, ZEROS padded to 288x288, tuned SSIM 
# Crop sinograms vertically to 288, then bilinearly resize horizontally to size 257.
# results in 288x257 size which is then padded with zeros horizontally to 288
config_ACT_SI= {
  "SI_alpha_min": -1,
  "SI_dropout": False,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": "Tanh",
  "SI_gen_hidden_dim": 10,
  "SI_gen_mult": 1.9886575078445312,
  "SI_gen_neck": "narrow",
  "SI_gen_z_dim": 1494,
  "SI_half_life_examples": -1,
  "SI_layer_norm": "none",
  "SI_learnedScale_init": 5.894328507562502,
  "SI_moment_1_fraction": -1,
  "SI_normalize": False,
  "SI_output_scale_lr_mult": 9.862580591354153,
  "SI_pad_mode": "replicate",
  "SI_skip_mode": "none",
  "SI_stats_criterion": -1,
  "batch_base2_exponent": 5,
  "gen_b1": 0.5583313727913738,
  "gen_b2": 0.3819733954553099,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.0016547324637469305,
  "gen_sino_channels": 3,
  "gen_sino_size": 288,
  "network_type": "ACT",
  "sup_base_criterion": "MSELoss",
  "train_SI": True
}
'''

'''
# (C, A) 288x288, tuned SSIM
# Crop sinograms vertically to 288, then average pool horizontally (pool size = 2)
# results in 288x257 size which is then padded with zeros horizontally to 288

config_ACT_SI = { 
  "SI_alpha_min": -1,
  "SI_dropout": False,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": "LeakyReLU",
  "SI_gen_hidden_dim": 10,
  "SI_gen_mult": 1.5012782419950113,
  "SI_gen_neck": "narrow", # narrow
  "SI_gen_z_dim": 872,
  "SI_half_life_examples": -1,
  "SI_layer_norm": "instance", # instance
  "SI_learnedScale_init": 7.305980552864529,
  "SI_normalize": False,
  "SI_output_scale_lr_mult": 1.6943444827125673,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "conv",
  "SI_stats_criterion": -1,
  "SI_moment_1_fraction": -1,
  "batch_base2_exponent": 7,
  "gen_b1": 0.3600790033157822,
  "gen_b2": 0.6033159868492163,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.0024018267054557695,
  "gen_sino_channels": 3,
  "gen_sino_size": 288, # 288
  "network_type": "ACT",
  "sup_base_criterion": "MSELoss",
  "train_SI": True
}
'''

###############################
### OLDER TUNINGS ARE BELOW ###
###############################
'''
## highCountSino-->actMap, tuned for SSIM, Augugment: SI
config_ACT_SI = {
  "SI_dropout": False,
  "SI_exp_kernel": 4, #4, 3
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": 'Sigmoid',
  "SI_gen_hidden_dim": 19,  # 19
  "SI_gen_mult": 2.065329728174869,
  "SI_gen_neck": "wide", #medium", "narrow", "wide"
  "SI_gen_z_dim": 1181,
  "SI_layer_norm": "none", # "instance", "group", "batch", "none"
  "SI_learnedScale_init": 4.2047521440377285,
  "SI_normalize": False,
  "SI_pad_mode": "zeros", # "zeros", "replicate"
  "SI_skip_mode": "add", # 'add', 'concat', 'none', 'conv'
  "batch_base2_exponent": 8, # 8 (my GPU can handle up to 9)
  "gen_b1": 0.22046050861804858,
  "gen_b2": 0.152643657443423,
  "gen_lr": 0.00099063275528607,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_sino_channels": 3, #3
  "gen_sino_size": 288, # default: 180 (288, 320)
  "sup_base_criterion": 'MSELoss',
  "SI_stats_criterion": 'PatchwiseMomentLoss',
  "SI_alpha_min": -1, # -1
  "SI_half_life_examples": 2000,
  "SI_output_scale_lr_mult": 1.0,  # No learnable output scale
  "network_type": "ACT",
  "train_SI": True,
  "SI_moment_1_fraction": -1
}
'''
'''
## highCountSino-->actMap, tuned for SSIM, Augugment: II
config_ACT_SI = {
   "SI_dropout": False,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 1,
  "SI_gen_final_activ": nn.ELU(alpha=1.0),
  "SI_gen_hidden_dim": 29,
  "SI_gen_mult": 1.5090047574838394,
  "SI_gen_neck": "wide",
  "SI_gen_z_dim": 486,
  "SI_layer_norm": "instance",
  "SI_learnedScale_init": 20.45467480669682,
  "SI_normalize": False,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "none",
  "batch_base2_exponent": 6,
  "gen_b1": 0.34632557248900636,
  "gen_b2": 0.10963336318792913,
  "gen_lr": 0.0005750756280291565,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "network_type": "ACT",
  "gen_sino_channels": 3,
  "gen_sino_size": 180,
  "sup_base_criterion": nn.MSELoss(),
  "sup_stats_criterion": None,
  "sup_alpha_min": 0.2,
  "sup_half_life_examples": 2000,
  "train_SI": True,
  "SI_moment_1_fraction": -1
}
'''

'''
## highCountImage-->actMap, tuned for SSIM, Augment: II (defective network)
config_ACT_SI = {
  "SI_dropout": True,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": nn.LeakyReLU(),
  "SI_gen_hidden_dim": 29,
  "SI_gen_mult": 3.378427521450207,
  "SI_gen_neck": "narrow", # 1 = smallest
  "SI_gen_z_dim": 2069,
  "SI_layer_norm": "group",
  "SI_learnedScale_init": 6.7836194674698165,
  "SI_normalize": False,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "concat", # concat
  "batch_base2_exponent": 6,
  "gen_b1": 0.42565713596651117,
  "gen_b2": 0.6898108744928462,
  "gen_lr": 0.0002493478013431121,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "network_type": "ACT",
  "gen_sino_channels": 1,
  "gen_sino_size": 180,
  "sup_base_criterion": nn.MSELoss(), # nn.MSELoss()
  "sup_stats_criterion": None,
  "sup_alpha_min": 0.2,
  "sup_half_life_examples": 2000,
  "train_SI": True,
  "SI_moment_1_fraction": -1
}
'''

###############################
### POORLY PERFORMING TRIAL, GOOD FOR DEBUGGING

'''
# Poorly performing network. Great for debugging.
config_ACT_SI = {
  "SI_alpha_min": 0.5524058308954609,
  "SI_dropout": True,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 1,
  "SI_gen_final_activ": "ELU",
  "SI_gen_hidden_dim": 16,
  "SI_gen_mult": 2.0064056851079384,
  "SI_gen_neck": "wide",
  "SI_gen_z_dim": 876,
  "SI_half_life_examples": 6748.004126224244,
  "SI_layer_norm": "group",
  "SI_learnedScale_init": 2.1259193148224687,
  "SI_moment_1_fraction": 0.49126420918366775,
  "SI_normalize": False,
  "SI_output_scale_lr_mult": 1.9980456276610286,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "none",
  "SI_stats_criterion": "PatchwiseMomentLoss",
  "batch_base2_exponent": 5,
  "gen_b1": 0.9659488635857614,
  "gen_b2": 0.2576282683921682,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.003606301749682214,
  "gen_sino_channels": 3,
  "gen_sino_size": 256,
  "network_type": "ACT",
  "sup_base_criterion": "VarWeightedMSE",
  "train_SI": True
}

'''