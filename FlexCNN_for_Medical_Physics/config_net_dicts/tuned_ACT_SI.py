from sympy import false
from torch import nn
from FlexCNN_for_Medical_Physics.classes.losses import PatchwiseMomentLoss, VarWeightedMSE

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

# Tried to stimulate Grokking
config_ACT_SI ={ # 256x256, bilinear intemediate size = 180, tuned for SSIM, pad_type='sinogram', fill enforced to =1
  "SI_alpha_min": -1,
  "SI_dropout": False,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 1,
  "SI_gen_final_activ": None,
  "SI_gen_hidden_dim": 15,
  "SI_gen_mult": 2.8577974008839018,
  "SI_gen_neck": "medium",
  "SI_gen_z_dim": 1824,
  "SI_half_life_examples": -1,
  "SI_layer_norm": "instance",
  "SI_learnedScale_init": 4.0327009143861074,
  "SI_normalize": False,
  "SI_output_scale_lr_mult": 2.9521606233259443,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "none",
  "SI_stats_criterion": -1,
  "batch_base2_exponent": 7,
  "gen_b1": 0.6207299459912765,
  "gen_b2": 0.5773966392271939,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.00026018975910785543,
  "gen_sino_channels": 3,
  "gen_sino_size": 256,
  "network_type": "ACT",
  "sup_base_criterion": "MSELoss",
  "train_SI": True
}

'''
# Wide sinogram pooling experiment
# NOTE: This set of hyperparameters was the basis for all the untuned experiments.
config_ACT_SI = { # 256x256, tuned SSIM, pad_type='sinogram', bilinear intermediate size = 180
  "SI_alpha_min": -1,
  "SI_dropout": False,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,  # 0
  "SI_gen_final_activ": None,
  "SI_gen_hidden_dim": 15,
  "SI_gen_mult": 3.3144387875060906,
  "SI_gen_neck": "medium",
  "SI_gen_z_dim": 1835,
  "SI_half_life_examples": -1,
  "SI_layer_norm": "group",
  "SI_learnedScale_init": 18.411171440894215,
  "SI_normalize": False,
  "SI_output_scale_lr_mult": 1.7607516297239543,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "none",
  "SI_stats_criterion": -1,
  "batch_base2_exponent": 6,
  "gen_b1": 0.17828464968859092,
  "gen_b2": 0.22254220083596676,
  "gen_image_channels": 1,
  "gen_image_size": 180,
  "gen_lr": 0.00011584402663085701,
  "gen_sino_channels": 3,
  "gen_sino_size": 256,
  "network_type": "ACT",
  "sup_base_criterion": "MSELoss",
  "train_SI": True
}
'''
'''
# Crop vertically, then pool horizontally experiment
# 288x288, tuned SSIM, pad_type='zeros', interemediate size = None, horiz_pool=2, vert_pool=1
# results in 257x257 size which is then padded with zeros to 288
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
'''
# Bilinear intermediate size to 161 (same pooling fraction  as 256x256 experiment
config_ACT_SI = { # 180x180, tuned SSIM, pad_type='zeros', bilinear_intermediate_size = 161
  "SI_alpha_min": -1,
  "SI_dropout": False,
  "SI_exp_kernel": 4,
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": "LeakyReLU",
  "SI_gen_hidden_dim": 11,
  "SI_gen_mult": 2.0282722914428213,
  "SI_gen_neck": "medium", # narrow
  "SI_gen_z_dim": 584,
  "SI_half_life_examples": -1,
  "SI_layer_norm": "instance",
  "SI_learnedScale_init": 10.553559972734485,
  "SI_normalize": False,
  "SI_output_scale_lr_mult": 8.240938610220685,
  "SI_pad_mode": "replicate",
  "SI_skip_mode": "conv",
  "SI_stats_criterion": -1,
  "batch_base2_exponent": 5,
  "gen_b1": 0.4495215605123463,
  "gen_b2": 0.15053718115803394,
  "gen_image_channels": 1,
  "gen_image_size": 180, # 180
  "gen_lr": 0.0003521542451328137,
  "gen_sino_channels": 3,
  "gen_sino_size": 256, # 256
  "network_type": "ACT",
  "sup_base_criterion": "MSELoss",
  "train_SI": True
}
'''

###############################
### OLDER TUNINGS ARE BELOW ###

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
  "train_SI": True
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
  "train_SI": True
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
  "train_SI": True
}
'''

'''
# 3x180x180 --> 1x180x180, Tuned for SSIM on OLDER dataset
config_ACT_SI = {
  "train_SI": True,
  "SI_dropout": False,
  "SI_exp_kernel": 4, # Options: 3, 4 (default 4)
  "SI_gen_fill": 0, # Options: 0,1,2 (default 0)
  "SI_gen_final_activ": nn.Tanh(),
  "SI_gen_hidden_dim": 23,
  "SI_gen_mult": 1.6605902406330195,
  "SI_gen_neck": "narrow", # Options: 1, 6, 11 (default 1)
  "SI_gen_z_dim": 789,
  "SI_layer_norm": "instance",
  "SI_pad_mode": "zeros",
  "batch_size": 71,
  "gen_b1": 0.2082092731474774,
  "gen_b2": 0.27147903136187507,
  "gen_lr": 0.0005481469822215635,
  "sup_base_criterion": nn.MSELoss(),
  "sup_stats_criterion": None,
  "sup_alpha_min": 0.2,
  "sup_half_life_examples": 2000,
  "network_type": "ACT",
  "gen_image_channels":1,
  "gen_sino_channels": 3,
  "gen_image_size":180,
  "gen_sino_size":180,
  "SI_normalize": True, # Default: 'true'
  "SI_fixedScale": 8100,
  "SI_skip_mode": "none", # Options: 'none', 'add', 'concat' (default 'none')
  }
'''