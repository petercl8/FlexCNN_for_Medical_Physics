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

### Below networks were tuned on whole dataset ###
# 1x90x90, Tuned for MSE - fc6 #


# 3x180x180 --> 1x180x180, Tuned for SSIM 
config_SUP_SI = {
  "train_SI": True,
  "SI_dropout": False,
  "SI_exp_kernel": 4,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": nn.Tanh(),
  "SI_gen_hidden_dim": 23,
  "SI_gen_mult": 1.6605902406330195,
  "SI_gen_neck": 1, # 1, 6, 11
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
  "SI_normalize": True,
  "SI_fixedScale": 8100,
  "SI_skip_mode": "concat", # 'none', 'add', 'concat'
  }

'''
# 3x90x90 --> 1x90x90, Tuned for SSIM 
config_SUP_SI = {
  "SI_dropout": False,
  "SI_exp_kernel": 4,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": nn.Tanh(),
  "SI_gen_hidden_dim": 23,
  "SI_gen_mult": 1.6605902406330195,
  "SI_gen_neck": 5,
  "SI_gen_z_dim": 789,
  "SI_layer_norm": "instance",
  "SI_normalize": True,
  "SI_pad_mode": "zeros",
  "batch_size": 71,
  "gen_b1": 0.2082092731474774,
  "gen_b2": 0.27147903136187507,
  "gen_lr": 0.0005481469822215635,
  "sup_criterion": nn.MSELoss(),
  "sino_size":90,
  "sino_channels": 3,
  "image_channels":1,
  "image_size":90,
  "train_SI": True, 
  "network_type": "SUP"
  "SI_fixedScale": 8100,
}
'''
'''
# 3x90x90-->1x90x90, Tuned for Local Distributions Metric, 10x10 window, stride 8 (b5c)
config_SUP_SI={
  "SI_dropout": False,
  "SI_exp_kernel": 4,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": None,
  "SI_gen_hidden_dim": 9,
  "SI_gen_mult": 2.1547197646081444,
  "SI_gen_neck": 5,
  "SI_gen_z_dim": 344,
  "SI_layer_norm": "batch",
  "SI_normalize": True,
  "SI_pad_mode": "zeros",
  "SI_fixedScale": 8100,
  "batch_size": 47,
  "gen_b1": 0.31108788447029295,
  "gen_b2": 0.3445239707919786,
  "gen_lr": 0.0007561178182660596,
  "sup_criterion": nn.L1Loss(),
  "sino_size":180,
  "sino_channels": 3,
  "image_channels":1,
  "image_size":90,
  "train_SI": True,
  "network_type": "SUP",
  "SI_fixedScale": 1,
}
'''
'''
# 1x90x90, Tuned for Local Distributions Metric, 5x5 window, stride 2
config_SUP_SI={
  "SI_dropout": True,
  "SI_exp_kernel": 3,
  "SI_gen_fill": 2,
  "SI_gen_final_activ": nn.Sigmoid(),
  "SI_gen_hidden_dim": 18,
  "SI_gen_mult": 2.4691388140182475,
  "SI_gen_neck": 11,
  "SI_gen_z_dim": 444,
  "SI_layer_norm": "instance",
  "SI_normalize": True,
  "SI_pad_mode": "zeros",
  "SI_fixedScale": 8100,
  "batch_size": 33,
  "gen_b1": 0.8199882799898334,
  "gen_b2": 0.1207854128656507,
  "gen_lr": 0.0001095057659925285,
  "sup_criterion": nn.BCEWithLogitsLoss()
  "sino_size":180,
  "sino_channels": 3,
  "image_channels":1,
  "image_size":90,
  "train_SI": True
}
'''