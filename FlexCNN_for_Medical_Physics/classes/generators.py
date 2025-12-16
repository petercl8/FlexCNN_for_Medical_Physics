import torch
from torch import nn

###########################
##### Generator Class #####
###########################

class Generator(nn.Module):
    def __init__(self, config, gen_SI=True):
        '''
        Encoder-decoder generator with optional skip connections, producing 180x180 output.
        Designed for domain transformation (e.g., sinogram->image, image->sinogram, or 
        iterative reconstruction->ground truth). Supports three skip connection modes:
        none (baseline), add (residual), and concat (U-Net style).

        Architecture: Contracting path (180->90->45->23->11), bottleneck neck (1, 6, or 11 spatial size),
        and expanding path (11->23->45->90->180). Skip connections cache encoder features at 90, 45, 23, 11
        and optionally merge them into the decoder at matching scales.

        Args:
            config: Dictionary containing network hyperparameters and data dimensions (sino_size, image_size,
                    sino_channels, image_channels). Direction-specific keys (SI_* for sinogram->image,
                    IS_* for image->sinogram) control network architecture:
                    - {SI,IS}_gen_neck: Bottleneck spatial size {1, 6, 11}
                    - {SI,IS}_exp_kernel: Expanding kernel size {3, 4}
                    - {SI,IS}_gen_z_dim: Channels in neck for neck=1 (dense bottleneck)
                    - {SI,IS}_gen_hidden_dim: Base channel count; all layers scale by mult**k
                    - {SI,IS}_gen_mult: Multiplicative factor for channels across layers
                    - {SI,IS}_gen_fill: Constant-size conv layers per block {0, 1, 2, 3}
                    - {SI,IS}_layer_norm: Normalization type {'batch', 'instance', 'group', 'none'}
                    - {SI,IS}_pad_mode: Padding mode {'zeros', 'reflect'}
                    - {SI,IS}_dropout: Whether to use dropout {True, False}
                    - {SI,IS}_skip_mode: Skip connection mode {'none', 'add', 'concat'}
                    - {SI,IS}_normalize: Whether to normalize output to L1 norm {True, False}
                    - {SI,IS}_fixedScale: Fixed output scaling factor (ignored if normalize=True)
                    - {SI,IS}_learnedScale_init: Initial value for learnable output scale
                    - {SI,IS}_gen_final_activ: Final activation {nn.Tanh(), nn.Sigmoid(), nn.ReLU(), None}
            gen_SI: Boolean - True for sinogram->image, False for image->sinogram. Controls which
                    config keys (SI_* vs IS_*) are used and input/output dimension assignment.
        '''
        super(Generator, self).__init__()

        if gen_SI:
            input_size = config['sino_size']
            input_channels = config['sino_channels']
            output_size = config['image_size']
            output_channels = config['image_channels']

            normalize_key = 'SI_normalize'
            fixed_key = 'SI_fixedScale'
            init_key = 'SI_learnedScale_init'
            skip_key = 'SI_skip_mode'

            neck = config['SI_gen_neck']
            exp_kernel = config['SI_exp_kernel']
            z_dim = config['SI_gen_z_dim']
            hidden_dim = config['SI_gen_hidden_dim']
            fill = config['SI_gen_fill']
            mult = config['SI_gen_mult']
            norm = config['SI_layer_norm']
            pad = config['SI_pad_mode']
            drop = config['SI_dropout']

            self.final_activation = config['SI_gen_final_activ']
            self.normalize = config['SI_normalize']
        else:
            input_size = config['image_size']
            input_channels = config['image_channels']
            output_size = config['sino_size']
            output_channels = config['sino_channels']

            normalize_key = 'IS_normalize'
            fixed_key = 'IS_fixedScale'
            init_key = 'IS_learnedScale_init'
            skip_key = 'IS_skip_mode'

            neck = config['IS_gen_neck']
            exp_kernel = config['IS_exp_kernel']
            z_dim = config['IS_gen_z_dim']
            hidden_dim = config['IS_gen_hidden_dim']
            fill = config['IS_gen_fill']
            mult = config['IS_gen_mult']
            norm = config['IS_layer_norm']
            pad = config['IS_pad_mode']
            drop = config['IS_dropout']

            self.final_activation = config['IS_gen_final_activ']
            self.normalize = config['IS_normalize']

        self.skip_mode = config.get(skip_key, 'none')

        self.output_scale_learnable = not bool(config.get(normalize_key, False)) # If learning scale only when not normalizing
        if self.output_scale_learnable:
            init_scale = float(config.get(init_key, config.get(fixed_key, 1.0)))
            self.log_output_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        else:
            init_scale = float(config.get(fixed_key, 1.0))
            self.register_buffer('fixed_output_scale', torch.tensor(init_scale, dtype=torch.float32))

        self.output_channels = output_channels
        self.output_size = output_size

        in_chan = input_channels
        out_chan = output_channels

        dim_0 = int(hidden_dim * mult**0)
        dim_1 = int(hidden_dim * mult**1)
        dim_2 = int(hidden_dim * mult**2)
        dim_3 = int(hidden_dim * mult**3)
        dim_4 = int(hidden_dim * mult**4)

        if input_size != 180:
            raise ValueError('This generator is configured for 180x180 inputs and outputs.')

        # Contracting Path: 180 -> 90 -> 45 -> 23 -> 11
        # Conv2d formula: H_out = floor((H_in + 2*padding - kernel) / stride) + 1
        self.contract_blocks = nn.ModuleList([
            contract_block(in_chan, dim_0, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (180+2-3)/2+1 = 90
            contract_block(dim_0, dim_1, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),    # H = (90+2-3)/2+1 = 45
            contract_block(dim_1, dim_2, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),    # H = (45+2-3)/2+1 = 23
            contract_block(dim_2, dim_2, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),    # H = (23+2-4)/2+1 = 11
        ])

        self.neck = self._build_neck(neck, dim_2, dim_3, dim_4, z_dim, pad, fill, norm, drop)
        self.expand_blocks = self._build_expand(exp_kernel, out_chan, dim_0, dim_1, dim_2, pad, fill, norm, drop)

    def _build_neck(self, neck, dim_2, dim_3, dim_4, z_dim, pad, fill, norm, drop):
        # neck=1: Narrowest bottleneck (1x1), upsamples back to 11x11 for skip merge
        if neck == 1:
            # ConvTranspose2d formula: H_out = (H_in-1)*stride + kernel - 2*padding + output_padding
            return nn.Sequential(
                contract_block(dim_2, dim_3, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (11+2-4)/2+1 = 5
                contract_block(dim_3, dim_4, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+2-3)/2+1 = 3
                contract_block(dim_4, z_dim, 3, stride=1, padding=0, padding_mode=pad, fill=0, norm='batch', drop=False),     # H = (3+0-3)/1+1 = 1
                expand_block(z_dim, dim_4, 3, stride=2, padding=0, output_padding=0, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (1-1)*2+3-0+0 = 3
                expand_block(dim_4, dim_3, 4, stride=2, padding=2, output_padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (3-1)*2+4-4+1 = 5
                expand_block(dim_3, dim_2, 3, stride=2, padding=0, output_padding=0, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (5-1)*2+3-0+0 = 11
            )

        # neck=6: Medium bottleneck (5x5 spatial), upsamples back to 11x11 for skip merge
        if neck == 5:
            return nn.Sequential(
                contract_block(dim_2, dim_3, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (11+2-4)/2+1 = 5
                contract_block(dim_3, dim_3, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                contract_block(dim_3, dim_3, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                contract_block(dim_3, dim_3, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                contract_block(dim_3, dim_3, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                expand_block(dim_3, dim_2, 3, stride=2, padding=0, output_padding=0, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (5-1)*2+3-0+0 = 11
            )

        # neck=11: Widest bottleneck (11x11 spatial), constant-size layers with kernel=5 for spatial information flow
        if neck == 11:
            return nn.Sequential(
                contract_block(dim_2, dim_2, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (11+4-5)/1+1 = 11 (constant)
                contract_block(dim_2, dim_2, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (11+4-5)/1+1 = 11 (constant)
                contract_block(dim_2, dim_2, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (11+4-5)/1+1 = 11 (constant)
                contract_block(dim_2, dim_2, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (11+4-5)/1+1 = 11 (constant)
                contract_block(dim_2, dim_2, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (11+4-5)/1+1 = 11 (constant)
            )

        raise ValueError('neck must be one of {1, 5, 11}')

    def _build_expand(self, exp_kernel, out_chan, dim_0, dim_1, dim_2, pad, fill, norm, drop):
        # Expanding Path: 11 -> 23 -> 45 -> 90 -> 180
        # ConvTranspose2d formula: H_out = (H_in-1)*stride + kernel - 2*padding + output_padding
        if exp_kernel == 3:
            stage_params = [
                (3, 2, 0, 0),  # 11 -> 23:  H = (11-1)*2+3-0+0 = 23
                (3, 2, 1, 0),  # 23 -> 45:  H = (23-1)*2+3-2+0 = 45
                (3, 2, 1, 1),  # 45 -> 90:  H = (45-1)*2+3-2+1 = 90
                (3, 2, 1, 1),  # 90 -> 180: H = (90-1)*2+3-2+1 = 180
            ]
        elif exp_kernel == 4:
            stage_params = [
                (4, 2, 1, 1),  # 11 -> 23:  H = (11-1)*2+4-2+1 = 23
                (4, 2, 2, 1),  # 23 -> 45:  H = (23-1)*2+4-4+1 = 45
                (4, 2, 1, 0),  # 45 -> 90:  H = (45-1)*2+4-2+0 = 90
                (4, 2, 1, 0),  # 90 -> 180: H = (90-1)*2+4-2+0 = 180
            ]
        else:
            raise ValueError('exp_kernel must be 3 or 4')

        def in_ch(base):
            return base * 2 if self.skip_mode == 'concat' else base

        blocks = nn.ModuleList()
        blocks.append(expand_block(in_ch(dim_2), dim_2, stage_params[0][0], stage_params[0][1], stage_params[0][2], stage_params[0][3], padding_mode=pad, fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_2), dim_1, stage_params[1][0], stage_params[1][1], stage_params[1][2], stage_params[1][3], padding_mode=pad, fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_1), dim_0, stage_params[2][0], stage_params[2][1], stage_params[2][2], stage_params[2][3], padding_mode=pad, fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_0), out_chan, stage_params[3][0], stage_params[3][1], stage_params[3][2], stage_params[3][3], padding_mode=pad, fill=fill, norm=norm, drop=drop, final_layer=True))
        return blocks

    def _merge(self, skip, x):
        if self.skip_mode == 'none' or skip is None:
            return x
        if self.skip_mode == 'add':
            return x + skip
        if self.skip_mode == 'concat':
            return torch.cat([x, skip], dim=1)
        raise ValueError('skip_mode must be one of {none, add, concat}')

    def forward(self, input):
        batch_size = len(input)

        skips = []
        a = input
        for block in self.contract_blocks:
            a = block(a)
            skips.append(a)

        a = self.neck(a)

        a = self._merge(skips[3], a)
        a = self.expand_blocks[0](a)

        a = self._merge(skips[2], a)
        a = self.expand_blocks[1](a)

        a = self._merge(skips[1], a)
        a = self.expand_blocks[2](a)

        a = self._merge(skips[0], a)
        a = self.expand_blocks[3](a)

        if self.final_activation:
            a = self.final_activation(a)
        if self.normalize:
            a = torch.reshape(a, (batch_size, self.output_channels, self.output_size**2))
            a = nn.functional.normalize(a, p=1, dim=2)
            a = torch.reshape(a, (batch_size, self.output_channels, self.output_size, self.output_size))

        scale = torch.exp(self.log_output_scale) if self.output_scale_learnable else self.fixed_output_scale
        a = a * scale
        return a


######################################
##### Block Generating Functions #####
######################################

def contract_block(in_channels, out_channels, kernel_size, stride, padding=0, padding_mode='reflect', fill=0, norm='batch', drop=False):
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d(out_channels)
    elif norm == 'instance':
        norm_layer = nn.InstanceNorm2d(out_channels)
    elif norm == 'group':
        num_groups = max(1, min(8, out_channels))
        while out_channels % num_groups != 0:
            num_groups -= 1
        norm_layer = nn.GroupNorm(num_groups, out_channels)
    else:
        norm_layer = nn.Sequential()

    dropout = nn.Dropout() if drop else nn.Sequential()

    block1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode),
        norm_layer,
        dropout,
        nn.ReLU(),
    )
    if fill == 0:
        block2 = nn.Sequential()
    elif fill == 1:
        block2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm_layer, dropout, nn.ReLU())
    elif fill == 2:
        block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm_layer, dropout, nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm_layer, dropout, nn.ReLU(),
        )
    elif fill == 3:
        block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm_layer, dropout, nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm_layer, dropout, nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode), norm_layer, dropout, nn.ReLU(),
        )
    else:
        block2 = nn.Sequential()
    return nn.Sequential(block1, block2)


def expand_block(in_channels, out_channels, kernel_size=3, stride=2, padding=0, output_padding=0, padding_mode='zeros', fill=0, norm='batch', drop=False, final_layer=False):
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d(out_channels)
    elif norm == 'instance':
        norm_layer = nn.InstanceNorm2d(out_channels)
    elif norm == 'group':
        num_groups = max(1, min(8, out_channels))
        while out_channels % num_groups != 0:
            num_groups -= 1
        norm_layer = nn.GroupNorm(num_groups, out_channels)
    else:
        norm_layer = nn.Sequential()

    dropout = nn.Dropout() if drop else nn.Sequential()

    block1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, padding_mode='zeros')
    if fill == 0:
        block2 = nn.Sequential()
    elif fill == 1:
        block2 = nn.Sequential(norm_layer, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode))
    elif fill == 2:
        block2 = nn.Sequential(
            norm_layer, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
            norm_layer, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
        )
    elif fill == 3:
        block2 = nn.Sequential(
            norm_layer, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
            norm_layer, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
            norm_layer, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
        )
    else:
        block2 = nn.Sequential()

    if not final_layer:
        block3 = nn.Sequential(norm_layer, dropout, nn.ReLU())
    else:
        block3 = nn.Sequential()

    return nn.Sequential(block1, block2, block3)
