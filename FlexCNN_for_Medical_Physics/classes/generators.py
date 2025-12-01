from torch import nn
import torch

###########################
##### Generator Class #####
###########################

class Generator(nn.Module):
    def __init__(self, config, gen_SI=True):
        '''
        A class to generate a 90x90-->90x90 or 180x180-->90x90 encoder-decoder network.
        The role of each item in the "config" dictionary is commented below.

        Args:
            config: Dictionary containing network hyperparameters and data dimensions (image_size, sino_size, 
                    image_channels, sino_channels). The constructor uses these to automatically determine 
                    input_size, input_channels, and output_channels based on gen_SI.
            gen_SI: Boolean - True if generating images from sinograms (SI), False if generating sinograms 
                    from images (IS). In a cycle-consistent network, this class generates two networks from 
                    the same config dictionary.
        '''
        
        super(Generator, self).__init__()
        
        # Derive input/output parameters from config based on gen_SI
        if gen_SI:  # Sinogram → Image
            input_size = config['sino_size']
            input_channels = config['sino_channels']
            output_size = config['image_size']
            output_channels = config['image_channels']
        else:  # Image → Sinogram
            input_size = config['image_size']
            input_channels = config['image_channels']
            output_size = config['sino_size']
            output_channels = config['sino_channels']

        # determine which scale key to read based on direction
        scale_key = 'SI' if gen_SI else 'IS'
        learnable_key = f'{scale_key}_output_scale_learnable'
        init_key = f'{scale_key}_output_scale_init'
        fixed_key = f'{scale_key}_scale'  # existing scale in your configs

        self.output_scale_learnable = bool(config.get(learnable_key, False))
        # prefer explicit init if provided, else fall back to existing fixed scale
        init_scale = float(config.get(init_key, config.get(fixed_key, 1.0)))

        if self.output_scale_learnable:
            # parameterize as log-scale so learned value stays positive and optimization is stable
            self.log_output_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        else:
            self.register_buffer('fixed_output_scale', torch.tensor(init_scale, dtype=torch.float32))


        ## Set Instance Variables ##
        self.output_channels = output_channels
        self.output_size = output_size
        ## If gen_SI == True, we use the "SI.." keys from the config dictionary to construct the generator network. ##
        if gen_SI:
            # The following instance variables are defined since these will be used in the forward() method below. #
            self.final_activation = config['SI_gen_final_activ']    # {nn.Tanh(), nn.Sigmoid(), None}
                                                                    # Type of activation function employed at the very end of network
            self.normalize=config['SI_normalize']                   # {True, False} : Normalization
            self.scale=config['SI_scale']                           # Scale factor by which the output is multiplied,
                                                                    #    if the output is first normalized

            ## The following variables are used in the network constructor, and not the forward() method, so there is no need for instance variables.

            neck=config['SI_gen_neck'] #            {1,5,11} :          Width of narrowest part (neck) of the network. The smaller the number, the narrower the neck.
            exp_kernel=config['SI_exp_kernel'] #    {3,4} :             Square kernel width (or height) for the expanding part of the network.
            z_dim=config['SI_gen_z_dim'] #          (Any real number) : Number of channels in the network neck, if neck=1. If neck=5 or 11, this parameter isn't used.
            hidden_dim=config['SI_gen_hidden_dim']# (Any real number) : scales all channels in network by the same linear factor. Larger hidden_dim -->more complex network
            fill=config['SI_gen_fill'] #            {0,1,2,3} :         Number of "constant size" 2D convolutions in each block
            mult=config['SI_gen_mult'] #            (Any real number) : Multiplicative factor by which network channels increase as the layers decrease in height & width
            norm=config['SI_layer_norm'] #          {'instance', 'batch', 'none'} : Type of layer normalization
            pad=config['SI_pad_mode'] #             {'zeros', 'reflect'} :          Type of padding in each layer/block
            drop=config['SI_dropout'] #             {'True', 'False'} :             Whether dropout is used in the network

        #If gen_SI == False, we use the "IS.." keys from the config dictionary to construct the generator network. ##
        else:
            self.final_activation = config['IS_gen_final_activ']
            self.normalize=config['IS_normalize']
            self.scale=config['IS_scale']

            neck=config['IS_gen_neck']
            exp_kernel=config['IS_exp_kernel']
            z_dim=config['IS_gen_z_dim']
            hidden_dim=config['IS_gen_hidden_dim']
            fill=config['IS_gen_fill']
            mult=config['IS_gen_mult']
            norm=config['IS_layer_norm']
            pad=config['IS_pad_mode']
            drop=config['IS_dropout']

        ## Abbreviations used for Block Definitions -- used to make code less awkward ##
        in_chan = input_channels
        out_chan = output_channels

        dim_0 = int(hidden_dim*mult**0) # Number of output channels of 1st block/input channels of 2nd block
        dim_1 = int(hidden_dim*mult**1) # Number of output channels of 2nd block/input channels of 3rd block
        dim_2 = int(hidden_dim*mult**2) # Follows pattern above
        dim_3 = int(hidden_dim*mult**3)
        dim_4 = int(hidden_dim*mult**4)
        dim_5 = int(hidden_dim*mult**5)

        ### Block Definitions ###

        ## Build the Contracting Path ##
        # The formula for the output size of a transposed convolution (nn.Conv2d) in Pytorch is as follows:
        # Hf = [Hi+2*padding-dilation(kernel-1)-1]/stride + 1 = [Hi+2*padding-kernel]/stride + 1 (for dialation=1)

        if input_size==180:
            self.contract = nn.Sequential(
                # nn.Conv2d: Hf = [Hi+2*padding-dilation(kernel-1)-1]/stride + 1 = [Hi+2*padding-kernel]/stride + 1 (for dialation=1)
                # Sinogram Shape: (3,90,90)
                contract_block(in_chan, dim_0, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop), # H = [180+2-3]/2 + 1 = 90
                contract_block(dim_0,   dim_1, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop), # H = [90+2-3]/2 + 1 = 45.5
                contract_block(dim_1,   dim_2, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop), # H = [45+2-3]/2 + 1 = 23
                contract_block(dim_2,   dim_2, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop), # H = [23+2-4]/2 + 1 = 11.5
            )
        elif input_size==90:
            self.contract = nn.Sequential(
                contract_block(in_chan, dim_0, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop), # H = [90+2-3]/2 + 1 = 45.5  : a 90x90 input gives a 45x45 output
                contract_block(dim_0,   dim_1, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop), # H = [45+2-3]/2 + 1 = 23    : a 45x45 input gives a 23x23 output
                contract_block(dim_1,   dim_2, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop), # H = [23+2-4]/2 + 1 = 11.5  : a 23x23 input gives a 11x11 output
            )

        ## Build the Neck. There are 3 options ##
        # neck=1 gives the narrowest (1x1) neck #
        if neck==1:
            self.neck = nn.Sequential(
                contract_block(dim_2, dim_3, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop), # H = [11+2-4]/2 + 1 = 5.5
                contract_block(dim_3, dim_4, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm           ), # H = [5+2*1-3]/2 + 1 = 3
                contract_block(dim_4, z_dim, 3, stride=1, padding=0,                   fill=0,    norm='batch'        ), # H = 1   ||norm is set to 'batch' because 'instance' won't work on 1x1 layer
                expand_block(  z_dim, dim_4, 3, stride=2, padding=0,                   fill=fill, norm=norm           ), # H = [1-1]*2+5 = 3
                expand_block(  dim_4, dim_3, 4, stride=2, padding=2, output_padding=1, fill=fill, norm=norm           ), # H = [3-1]*2+4-2*2+1 = 5
            )

        # neck=5 gives the middle width (5x5) neck #
        if neck==5:
            self.neck = nn.Sequential(
                contract_block(dim_2, dim_3, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop), # H = [11+2-4]/2 + 1 = 5.5
                contract_block(dim_3, dim_3, 5, stride=1, padding=2, padding_mode=pad,            norm=norm           ), # H = [5+2*2-5]/1 + 1 = 5 (Constant Block)
                contract_block(dim_3, dim_3, 5, stride=1, padding=2, padding_mode=pad,            norm=norm           ), # H = [5+2*2-5]/1 + 1 = 5 (Constant Block)
                contract_block(dim_3, dim_3, 5, stride=1, padding=2, padding_mode=pad,            norm=norm           ), # H = [5+2*2-5]/1 + 1 = 5 (Constant Block)
                #contract_block(dim_3, dim_3, kernel_size=5, stride=1, padding=2, padding_mode=pad, norm=norm), # H = [5+2*2-5]/1 + 1 = 5 (Constant Block) # Add this next tuning!
            )

        # neck=11 gives the thickest (11x11) neck #
        if neck==11:
            self.neck = nn.Sequential(
                contract_block(dim_2, dim_2, kernel_size=5, stride=1, padding=2, padding_mode=pad, norm=norm), # H = [11+2*2-5]/1 + 1 = 11 (Constant Block)
                contract_block(dim_2, dim_2, kernel_size=5, stride=1, padding=2, padding_mode=pad, norm=norm), # H = [11+2*2-5]/1 + 1 = 11 (Constant Block)
                contract_block(dim_2, dim_2, kernel_size=5, stride=1, padding=2, padding_mode=pad, norm=norm), # H = [11+2*2-5]/1 + 1 = 11 (Constant Block)
                contract_block(dim_2, dim_2, kernel_size=5, stride=1, padding=2, padding_mode=pad, norm=norm), # H = [11+2*2-5]/1 + 1 = 11 (Constant Block)
                contract_block(dim_2, dim_2, kernel_size=5, stride=1, padding=2, padding_mode=pad, norm=norm), # H = [11+2*2-5]/1 + 1 = 11 (Constant Block)
            )

        ## Build the Expanding Blocks ##
        # The formula for the output size of a transposed convolution (nn.ConvTranspose2d:) in Pytorch is as follows:
        # Hf = (Hi-1)*stride -2*padding +dilation*(kernel-1) +output_padding+1
        #    = (Hi-1)*stride +kernel -2*padding +output_padding (for dialation=1)

        # For neck=1 or 5, the output from previous layers is 5x5. Therefore, these can use the same expanding blocks #
        if (neck==1 or neck==5):
            if exp_kernel==3:
            # Expanding block for neck=1 or 5, expanding kernel size = 3)
                self.expand = nn.Sequential(
                    expand_block(dim_3, dim_2,                      kernel_size=3, stride=2, padding=0, output_padding=0, fill=fill, norm=norm), # H = (5-1)*2  +3         = 11
                    expand_block(dim_2, dim_1,                      kernel_size=3, stride=2, padding=1, output_padding=1, fill=fill, norm=norm), # H = (11-1)*2 +3 -2*1 +1 = 22
                    expand_block(dim_1, dim_0,                      kernel_size=3, stride=2, padding=0, output_padding=0, fill=fill, norm=norm), # H = (22-1)*2 +3         = 45
                    expand_block(dim_0, out_chan, final_layer=True, kernel_size=3, stride=2, padding=1, output_padding=1, fill=fill, norm=norm), # H = (45-1)*2 +3 -2*1 +1 = 90
                )

            elif exp_kernel==4:
            # Expanding block for neck=1 or 5, expanding kernel size = 4
                self.expand = nn.Sequential(
                    expand_block(dim_3, dim_2,                      kernel_size=4, stride=2, padding=1, output_padding=1, fill=fill, norm=norm),  # H = (5-1)*2  +4 -2*1 +1 = 11
                    expand_block(dim_2, dim_1,                      kernel_size=4, stride=2, padding=1, output_padding=0, fill=fill, norm=norm),  # H = (11-1)*2 +4 -2*1    = 22
                    expand_block(dim_1, dim_0,                      kernel_size=4, stride=2, padding=1, output_padding=1, fill=fill, norm=norm),  # H = (21-1)*2 +4 -2*1 +1 = 45
                    expand_block(dim_0, out_chan, final_layer=True, kernel_size=4, stride=2, padding=1, output_padding=0, fill=fill, norm=norm),  # H = (45-1)*2 +4 -2*1    = 90
                )

        # For neck=11, the output is 11x11. This neck requires its own expanding blocks #
        if neck==11:
            if exp_kernel==3:
            # Expanding block for neck=11, expanding kernel size = 3
                self.expand = nn.Sequential(
                    expand_block(dim_2, dim_1,                      kernel_size=3, stride=2, padding=1, output_padding=1, fill=fill, norm=norm),  # H = (11-1)*2 +3 -2*1 +1 = 22
                    expand_block(dim_1, dim_0,                      kernel_size=3, stride=2, padding=0, output_padding=0, fill=fill, norm=norm),  # H = (22-1)*2 +3         = 45
                    expand_block(dim_0, out_chan, final_layer=True, kernel_size=3, stride=2, padding=1, output_padding=1, fill=fill, norm=norm),  # H = (45-1)*2 +3 -2*1 +1 = 90
                )

            if exp_kernel==4:
            # Expanding block for neck=11, expanding kernel size = 4
                self.expand = nn.Sequential(
                    expand_block(dim_2, dim_1,                      kernel_size=4, stride=2, padding=1, output_padding=0, fill=fill, norm=norm),  # H = (11-1)*2 +4 -2*1    = 22
                    expand_block(dim_1, dim_0,                      kernel_size=4, stride=2, padding=1, output_padding=1, fill=fill, norm=norm),  # H = (21-1)*2 +4 -2*1 +1 = 45
                    expand_block(dim_0, out_chan, final_layer=True, kernel_size=4, stride=2, padding=1, output_padding=0, fill=fill, norm=norm),  # H = (45-1)*2 +4 -2*1    = 90
                )

    def forward(self, input):
        # This method gets run when the network is called to produce an output from an input #

        batch_size = len(input)  # Get batch size

        a = self.contract(input) # Run input through contracting blocks
        a = self.neck(a)         # Run output from contracting blocks through the neck
        a = self.expand(a)       # Run outoput from the neck through the expanding blocks

        if self.final_activation:   # Optional final activations
            a = self.final_activation(a)
        if self.normalize:          # Optionally normalize
            a = torch.reshape(a,(batch_size, self.output_channels, self.output_size**2)) # Flattens each image
            a = nn.functional.normalize(a, p=1, dim = 2)
            a = torch.reshape(a,(batch_size, self.output_channels , self.output_size, self.output_size)) # Reshapes images back into square matrices

        if self.output_scale_learnable:
            scale = torch.exp(self.log_output_scale)
        else:
            scale = self.fixed_output_scale

        a = a * scale
        return a                    # Return the output
    



######################################
##### Block Generating Functions #####
######################################

def contract_block(in_channels, out_channels, kernel_size, stride, padding=0, padding_mode='reflect', fill=0, norm='batch', drop=False):
    '''
    Function to construct a single "contracting block." Each contracting block consists of one 2D convolutional layer, which decreases
    the size (height and width) of the data. There are then up to three 2D convolution layers which do not change the height or width
    (e.g. "constant size layers").

    in_channels:    number of channels at the input of contracting block
    out_channels:   number of channels at the output of contracting block
    kernel_size:    size of the kernel for the 1st 2D convolutional layer in the contracting block
    stride:         stride of the convolution for the 1st 2D convolutional layer in the contracting block
    padding:        amount of padding for the the 1st 2D convolutional layer in the contracting block
    padding_mode:   padding mode (options: "zeros", "reflect")
    fill:           number of "constant size" 2D convolutional layers
    norm:           type of layer normalization ("batch", "instance", or "none")
    dropout:        include dropout layers in the contracting block? (True or False)
    '''

    if norm=='batch':    norm = nn.BatchNorm2d(out_channels)
    if norm=='instance': norm = nn.InstanceNorm2d(out_channels)
    if norm=='none':     norm = nn.Sequential()
    dropout = nn.Dropout() if drop==True else nn.Sequential()

    # Note: for the contracting block, normalization & dropout follow convolutional layers. For expanding blocks, the order is reversed.
    block1 =  nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode), norm, dropout, nn.ReLU())
    if fill==0:
        block2 = nn.Sequential() # If fill=0, there are no "constant size" convolutional layers, and so block2 is empty.
    if fill==1:
        block2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm, dropout, nn.ReLU())
    elif fill==2:
        block2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm, dropout, nn.ReLU(),
                                nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm, dropout, nn.ReLU())
    elif fill==3:
        block2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm, dropout, nn.ReLU(),
                                nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm, dropout, nn.ReLU(),
                                nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm, dropout, nn.ReLU())
    return nn.Sequential(block1, block2)

def expand_block(in_channels, out_channels, kernel_size=3, stride=2, padding=0, output_padding=0, padding_mode='zeros', fill=0, norm='batch', drop=False, final_layer=False):
    '''
    Function to construct a single "expanding block." Each expanding block consists of one 2D transposed convolution layer which increases
    the size of the incoming data (height and width). There are then up to three 2D convolution layers which do not change the height or
    width (e.g. "constant size layers").

    in_channels:    number of channels at the input of the expanding block
    out_channels:   number of channels at the output of the expanding block
    kernel_size:    size of the kernel for the 1st 2D transposed convolutional layer in the expanding block
    stride:         stride of the convolution for the 1st 2D transposed convolutional layer in the expanding block
    padding:        amount of padding for the the 1st 2D transposed convolutional layer in the expanding block
    padding_mode:   padding mode (ex: "zeros", "reflect")
    fill:           number of "constant size" 2D convolutional layers
    norm:           type of layer normalization ("batch", "instance", or "none")
    dropout:        include dropout in the expanding block (True or False)
    final_layer:    Is this the final layer in the expanding block? (True or False)
    '''

    if norm=='batch':       norm = nn.BatchNorm2d(out_channels)
    if norm=='instance':    norm = nn.InstanceNorm2d(out_channels)
    if norm=='none':        norm = nn.Sequential()
    dropout = nn.Dropout() if drop==True else nn.Sequential()

    # Note: for the expanding block, normalization & dropout precede convolutional layers in blocks 2-3. For expanding blocks, the order is reversed.
    block1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, padding_mode=padding_mode)
    if fill==0:
        block2 = nn.Sequential()
    if fill==1:
        block2 = nn.Sequential(norm, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode))
    elif fill==2: # For
        block2 = nn.Sequential(norm, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
                                norm, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode))
    elif fill==3:
        block2 = nn.Sequential(norm, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
                                norm, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
                                norm, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode))

    if final_layer==False: # If not the final layer, I add normalization, dropout and activation.
        block3 = nn.Sequential(norm, dropout, nn.ReLU())
    else:                  # Otherwise, I leave off the normalization, dropout, and activation. This allows me to do it explicitly
                           # at the end of the network using tuned parameters.
        block3 = nn.Sequential()
    return nn.Sequential(block1, block2, block3)