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

## highCountImage-->actMap, tuned for SSIM
config_SUP_SI = {
  "SI_dropout": True,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": nn.LeakyReLU(),
  "SI_gen_hidden_dim": 29,
  "SI_gen_mult": 3.378427521450207,
  "SI_gen_neck": 11, # 1 = smallest
  "SI_gen_z_dim": 2069,
  "SI_layer_norm": "group",
  "SI_learnedScale_init": 6.7836194674698165,
  "SI_normalize": False,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "concat",
  "batch_base2_exponent": 6,
  "gen_b1": 0.42565713596651117,
  "gen_b2": 0.6898108744928462,
  "gen_lr": 0.0002493478013431121,
  "image_channels": 1,
  "image_size": 180,
  "network_type": "SUP",
  "sino_channels": 1,
  "sino_size": 180,
  "sup_criterion": nn.MSELoss(),
  "train_SI": True
}

'''
## highCountSino-->actMap, tuned for SSIM
config_SUP_SI = {
  "SI_dropout": False,
  "SI_exp_kernel": 3,
  "SI_fixedScale": 1,
  "SI_gen_fill": 1,
  "SI_gen_final_activ": nn.LeakyReLU(),
  "SI_gen_hidden_dim": 8,
  "SI_gen_mult": 2.003683891151235,
  "SI_gen_neck": 11,
  "SI_gen_z_dim": 258,
  "SI_layer_norm": "instance",
  "SI_learnedScale_init": 0.8270848915353621,
  "SI_normalize": False,
  "SI_pad_mode": "zeros",
  "SI_skip_mode": "concat",
  "batch_base2_exponent": 5,
  "gen_b1": 0.389118911125988,
  "gen_b2": 0.2351938098613643,
  "gen_lr": 0.00220256446257333,
  "image_channels": 1,
  "image_size": 180,
  "network_type": "SUP",
  "sino_channels": 3,
  "sino_size": 180,
  "sup_criterion": nn.MSELoss(),
  "train_SI": True
}
'''

'''
# 3x180x180 --> 1x180x180, Tuned for SSIM on OLDER dataset
config_SUP_SI = {
  "train_SI": True,
  "SI_dropout": False,
  "SI_exp_kernel": 4, # Options: 3, 4 (default 4)
  "SI_gen_fill": 0, # Options: 0,1,2 (default 0)
  "SI_gen_final_activ": nn.Tanh(),
  "SI_gen_hidden_dim": 23,
  "SI_gen_mult": 1.6605902406330195,
  "SI_gen_neck": 1, # Options: 1, 6, 11 (default 1)
  "SI_gen_z_dim": 789,
  "SI_layer_norm": "instance",
  "SI_pad_mode": "zeros",
  "batch_size": 71,
  "gen_b1": 0.2082092731474774,
  "gen_b2": 0.27147903136187507,
  "gen_lr": 0.0005481469822215635,
  "sup_criterion": nn.MSELoss(),
  "network_type": "SUP",
  "image_channels":1,
  "sino_channels": 3,
  "image_size":180,
  "sino_size":180,
  "SI_normalize": True, # Default: 'true'
  "SI_fixedScale": 8100,
  "SI_skip_mode": "none", # Options: 'none', 'add', 'concat' (default 'none')
  }
'''