import torch
from torch import nn


#############################
##### Generator Classes #####
#############################

class Generator_288(nn.Module):
    def __init__(self, config, gen_SI=True, gen_skip_handling: str = 'classic', gen_flow_mode: str = 'coflow', enc_inject_channels=None, dec_inject_channels=None):
        '''
        Encoder-decoder generator with optional skip connections, producing 288x288 output.
        Contracting path: 288->144->72->36->18->9, neck: 1x1, 5x5, or 9x9, expanding path: 9->18->36->72->144->288.
        Skip handling modes:
            - classic: standard U-Net add/concat/none skips (no injection).
            - 1x1Conv: skip + optional frozen features are concatenated then reduced via 1x1 conv at decoder stages.
        Injection channel tuples (enc/dec) are passed explicitly (order: 144, 36, 9); use None or zeros to disable.
        '''
        super(Generator_288, self).__init__()

        if gen_SI:
            input_size = config['gen_sino_size']
            input_channels = config['gen_sino_channels']
            output_size = config['gen_image_size']
            output_channels = config['gen_image_channels']

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
            input_size = config['gen_image_size']
            input_channels = config['gen_image_channels']
            output_size = config['gen_sino_size']
            output_channels = config['gen_sino_channels']

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

        # Skip handling and injection configuration (architecture-level)
        self.skip_handling = gen_skip_handling
        if self.skip_handling not in ('classic', '1x1Conv'):
            raise ValueError('gen_skip_handling must be one of {classic, 1x1Conv}')

        self.flow_mode = gen_flow_mode
        if self.flow_mode not in ('coflow', 'counterflow'):
            raise ValueError('gen_flow_mode must be one of {coflow, counterflow}')

        def _normalize_tuple(cfg):
            if cfg is None:
                return (0, 0, 0)
            if len(cfg) != 3:
                raise ValueError('Injection tuples must have three entries (144,36,9 scales).')
            return tuple(int(x) for x in cfg)

        self.enc_inject_channels = _normalize_tuple(enc_inject_channels)
        self.dec_inject_channels = _normalize_tuple(dec_inject_channels)
        if self.skip_handling == 'classic' and (any(self.enc_inject_channels) or any(self.dec_inject_channels)):
            raise ValueError('Injection requires gen_skip_handling="1x1Conv"')
        self.enable_encoder_inject = self.skip_handling == '1x1Conv' and any(self.enc_inject_channels)
        self.enable_decoder_inject = self.skip_handling == '1x1Conv'

        self.output_scale_learnable = not bool(config.get(normalize_key, False))
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

        # Root scaling exponent for channel growth
        self.scaling_exp = 0.7  # Change as needed (e.g., 0.7)
        # Unique channel dims for each stage
        dim_0 = int(hidden_dim * mult ** (0 ** self.scaling_exp))
        dim_1 = int(hidden_dim * mult ** (1 ** self.scaling_exp))
        dim_2 = int(hidden_dim * mult ** (2 ** self.scaling_exp))
        dim_3 = int(hidden_dim * mult ** (3 ** self.scaling_exp))
        dim_4 = int(hidden_dim * mult ** (4 ** self.scaling_exp))
        dim_5 = int(hidden_dim * mult ** (5 ** self.scaling_exp))
        dim_6 = int(hidden_dim * mult ** (6 ** self.scaling_exp))

        if input_size != 288:
            raise ValueError('This generator is configured for 288x288 inputs.')

        # Contracting Path: 288 -> 144 -> 72 -> 36 -> 18 -> 9
        self.contract_blocks = nn.ModuleList([
            contract_block(in_chan, dim_0, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),   # 288->144
            contract_block(dim_0, dim_1, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 144->72
            contract_block(dim_1, dim_2, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 72->36
            contract_block(dim_2, dim_3, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 36->18
            contract_block(dim_3, dim_4, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 18->9
        ])

        self.neck = self._build_neck(neck, dim_4, dim_5, dim_6, z_dim, pad, fill, norm, drop)
        self.expand_blocks = self._build_expand(exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, dim_4, pad, fill, norm, drop, self.skip_handling)

        # Channel references for injection (144,36,9 scales)
        self.enc_stage_channels = (dim_0, dim_2, dim_4)
        # Decoder stages at resolutions 9 (pre first upsample), 36 (pre 36->144 upsample), 144 (pre final upsample)
        self.dec_stage_channels = (dim_0, dim_2, dim_4)
        self.dec_skip_channels = (dim_0, dim_2, dim_4)

        if self.skip_handling == '1x1Conv':
            self._build_injectors(pad, norm, drop)

    def _build_neck(self, neck, dim_4, dim_5, dim_6, z_dim, pad, fill, norm, drop):
        # neck='narrow': 1x1 bottleneck
        if neck == 'narrow':
            return nn.Sequential(
                contract_block(dim_4, dim_5, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # 9->5: floor((9+2-3)/2)+1=5
                contract_block(dim_5, dim_6, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # 5->3: floor((5+2-3)/2)+1=3
                contract_block(dim_6, z_dim, 3, stride=1, padding=0, padding_mode=pad, fill=0, norm='batch', drop=False),     # 3->1
                expand_block(z_dim, dim_6, 3, stride=2, padding=0, output_padding=0, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # 1->3
                expand_block(dim_6, dim_5, 4, stride=2, padding=2, output_padding=1, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # 3->5
                expand_block(dim_5, dim_4, 3, stride=2, padding=1, output_padding=0, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # 5->9: (5-1)*2+3-2+0=9
            )
        # neck='medium': 5x5 bottleneck
        if neck == 'medium':
            return nn.Sequential(
                contract_block(dim_4, dim_5, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # 9->5: floor((9+2-3)/2)+1=5
                contract_block(dim_5, dim_5, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # 5->5
                contract_block(dim_5, dim_5, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # 5->5
                contract_block(dim_5, dim_5, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # 5->5
                contract_block(dim_5, dim_5, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # 5->5
                expand_block(dim_5, dim_4, 3, stride=2, padding=1, output_padding=0, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # 5->9: (5-1)*2+3-2+0=9
            )
        # neck='wide': 9x9 bottleneck
        if neck == 'wide':
            return nn.Sequential(
                contract_block(dim_4, dim_4, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # 9->9
                contract_block(dim_4, dim_4, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # 9->9
                contract_block(dim_4, dim_4, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # 9->9
                contract_block(dim_4, dim_4, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # 9->9
                contract_block(dim_4, dim_4, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # 9->9
            )
        raise ValueError('neck must be one of {narrow, medium, wide} for Generator_288')

    def _build_expand(self, exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, dim_4, pad, fill, norm, drop, skip_handling):
        # Expanding Path: 9 -> 18 -> 36 -> 72 -> 144 -> 288
        if exp_kernel == 3:
            stage_params = [
                (3, 2, 1, 1),  # 9->18
                (3, 2, 1, 1),  # 18->36
                (3, 2, 1, 1),  # 36->72
                (3, 2, 1, 1),  # 72->144
                (3, 2, 1, 1),  # 144->288
            ]
        elif exp_kernel == 4:
            stage_params = [
                (4, 2, 1, 0),  # 9->18
                (4, 2, 1, 0),  # 18->36
                (4, 2, 1, 0),  # 36->72
                (4, 2, 1, 0),  # 72->144
                (4, 2, 1, 0),  # 144->288
            ]
        else:
            raise ValueError('exp_kernel must be 3 or 4')

        def in_ch(base):
            if skip_handling == '1x1Conv':
                return base
            return base * 2 if self.skip_mode == 'concat' else base

        blocks = nn.ModuleList()
        blocks.append(expand_block(in_ch(dim_4), dim_3, stage_params[0][0], stage_params[0][1], stage_params[0][2], stage_params[0][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_3), dim_2, stage_params[1][0], stage_params[1][1], stage_params[1][2], stage_params[1][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_2), dim_1, stage_params[2][0], stage_params[2][1], stage_params[2][2], stage_params[2][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_1), dim_0, stage_params[3][0], stage_params[3][1], stage_params[3][2], stage_params[3][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_0), out_chan, stage_params[4][0], stage_params[4][1], stage_params[4][2], stage_params[4][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop, final_layer=True))
        return blocks

    def _build_injectors(self, pad, norm, drop):
        def _make_proj(in_ch, out_ch):
            return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        # Encoder injectors (only if requested)
        enc_keys = ['enc_144', 'enc_36', 'enc_9']
        enc_chs = self.enc_stage_channels
        self.enc_injectors = nn.ModuleDict()
        for key, base_ch, inj_ch in zip(enc_keys, enc_chs, self.enc_inject_channels):
            if inj_ch > 0:
                self.enc_injectors[key] = _make_proj(base_ch + inj_ch, base_ch)

        # Decoder injectors (always in 1x1Conv mode)
        dec_keys = ['dec_144', 'dec_36', 'dec_9']
        dec_chs = self.dec_stage_channels
        skip_chs = self.dec_skip_channels
        self.dec_injectors = nn.ModuleDict()
        for key, base_ch, skip_ch, inj_ch in zip(dec_keys, dec_chs, skip_chs, self.dec_inject_channels):
            total_in = base_ch
            if self.skip_mode != 'none':
                total_in += skip_ch
            total_in += inj_ch
            self.dec_injectors[key] = _make_proj(total_in, base_ch)

    def _merge(self, skip, x):
        if self.skip_mode == 'none' or skip is None:
            return x
        if self.skip_mode == 'add':
            return x + skip
        if self.skip_mode == 'concat':
            return torch.cat([x, skip], dim=1)
        raise ValueError('skip_mode must be one of {none, add, concat}')

    def forward(self, input, frozen_encoder_features=None, frozen_decoder_features=None, return_features: bool = False):
        batch_size = len(input)

        # ============================================================================
        # SETUP AND VALIDATION
        # ============================================================================
        if self.skip_handling == 'classic' and (frozen_encoder_features is not None or frozen_decoder_features is not None):
            raise ValueError('Frozen features provided but gen_skip_handling is classic.')

        # Route frozen features based on flow mode (coflow: unchanged, counterflow: swap)
        enc_feats_in = frozen_encoder_features
        dec_feats_in = frozen_decoder_features
        if self.flow_mode == 'counterflow':
            enc_feats_in, dec_feats_in = dec_feats_in, enc_feats_in

        enc_feats_in = tuple(enc_feats_in) if enc_feats_in is not None else None
        dec_feats_in = tuple(dec_feats_in) if dec_feats_in is not None else None

        if self.skip_handling == '1x1Conv':
            if any(self.enc_inject_channels) and enc_feats_in is None:
                raise ValueError('Encoder injection requested but frozen_encoder_features not provided.')
            if self.enable_decoder_inject and dec_feats_in is None and any(self.dec_inject_channels):
                raise ValueError('Decoder injection requested but frozen_decoder_features not provided.')

        def _assert_match(tensor, target_h, target_w, expected_c, label):
            if tensor is None:
                raise ValueError(f'Missing tensor for {label}')
            h, w = tensor.shape[-2:]
            if h != target_h or w != target_w:
                raise ValueError(f'Shape mismatch for {label}: expected {target_h}x{target_w}, got {h}x{w}')
            if tensor.shape[1] != expected_c:
                raise ValueError(f'Channel mismatch for {label}: expected {expected_c}, got {tensor.shape[1]}')

        # ============================================================================
        # ENCODER: Contraction 288 -> 144 -> 72 -> 36 -> 18 -> 9 with injection
        # ============================================================================
        skips = []
        a = input
        for idx, block in enumerate(self.contract_blocks):
            a = block(a)
            # Inject frozen features at scales 144 (idx=0), 36 (idx=2), 9 (idx=4)
            if self.skip_handling == '1x1Conv' and idx in (0, 2, 4):
                inj_idx = {0: 0, 2: 1, 4: 2}[idx]
                inj_ch = self.enc_inject_channels[inj_idx]
                inj_feat = enc_feats_in[inj_idx]
                _assert_match(inj_feat, a.shape[-2], a.shape[-1], inj_ch, f'encoder_inject_{inj_idx}')
                key = ('enc_144', 'enc_36', 'enc_9')[inj_idx]
                a = torch.cat([a, inj_feat], dim=1) # Concatenate along channel dimension
                a = self.enc_injectors[key](a)      # Project back to base channels
            skips.append(a)

        if return_features:
            encoder_feats = [skips[0], skips[2], skips[4]]  # 144, 36, 9

        # ============================================================================
        # BOTTLENECK
        # ============================================================================
        a = self.neck(a)

        # ============================================================================
        # DECODER: Expansion with injection at three stages
        # ============================================================================

        # --- DECODER STAGE 1: 9x9 -> 18x18 ---
        if self.skip_handling == '1x1Conv':
            inj_feat = dec_feats_in[2] if dec_feats_in is not None else None
            inj_ch = self.dec_inject_channels[2]
            _assert_match(inj_feat, skips[4].shape[-2], skips[4].shape[-1], inj_ch, 'decoder_inject_9')
            parts = [a] # Current decoder features
            if self.skip_mode != 'none':
                parts.append(skips[4])         # If you are using skips, add the skip connection channels
            parts.append(inj_feat)             # Add the injection features
            a = torch.cat(parts, dim=1)        # Concatenate along channel dimension
            a = self.dec_injectors['dec_9'](a) # Project back to base channels
            if return_features:
                decoder_feat_9 = a             # Store features before upsample
        else:
            a = self._merge(skips[4], a)
            if return_features:
                decoder_feat_9 = a

        a = self.expand_blocks[0](a)  # 9 -> 18

        # --- INTERMEDIATE STAGES: 18/36 (classic mode only) ---
        if self.skip_handling == 'classic':
            a = self._merge(skips[3], a)
        a = self.expand_blocks[1](a)  # 18 -> 36

        if self.skip_handling == 'classic':
            a = self._merge(skips[2], a)
        a = self.expand_blocks[2](a)  # 36 -> 72

        # --- DECODER STAGE 2: 36x36 -> 144x144 ---
        if self.skip_handling == '1x1Conv':
            inj_feat = dec_feats_in[1] if dec_feats_in is not None else None
            inj_ch = self.dec_inject_channels[1]
            _assert_match(inj_feat, skips[2].shape[-2], skips[2].shape[-1], inj_ch, 'decoder_inject_36')
            parts = [a]
            if self.skip_mode != 'none':
                parts.append(skips[2])
            parts.append(inj_feat)
            a = torch.cat(parts, dim=1)
            a = self.dec_injectors['dec_36'](a)
            if return_features:
                decoder_feat_36 = a
        else:
            if return_features:
                decoder_feat_36 = a
            a = self._merge(skips[1], a)

        a = self.expand_blocks[3](a)  # 72 -> 144

        # --- DECODER STAGE 3: 144x144 (before final upsample) ---
        if self.skip_handling == '1x1Conv':
            inj_feat = dec_feats_in[0] if dec_feats_in is not None else None
            inj_ch = self.dec_inject_channels[0]
            _assert_match(inj_feat, skips[0].shape[-2], skips[0].shape[-1], inj_ch, 'decoder_inject_144')
            parts = [a]
            if self.skip_mode != 'none':
                parts.append(skips[0])
            parts.append(inj_feat)
            a = torch.cat(parts, dim=1)
            a = self.dec_injectors['dec_144'](a)
            if return_features:
                decoder_feat_144 = a
        else:
            if return_features:
                decoder_feat_144 = a
            a = self._merge(skips[0], a)

        a = self.expand_blocks[4](a)  # 144 -> 288

        # ============================================================================
        # POST-PROCESSING: Cropping, activation, normalization, scaling
        # ============================================================================
        if a.shape[-1] > self.output_size:
            crop_size = self.output_size
            margin = (a.shape[-1] - crop_size) // 2
            a = a[:, :, margin:margin+crop_size, margin:margin+crop_size]

        if self.final_activation:
            a = self.final_activation(a)

        if self.normalize:
            a = torch.reshape(a, (batch_size, self.output_channels, self.output_size**2))
            a = nn.functional.normalize(a, p=1, dim=2)
            a = torch.reshape(a, (batch_size, self.output_channels, self.output_size, self.output_size))

        scale = torch.exp(self.log_output_scale) if self.output_scale_learnable else self.fixed_output_scale
        a = a * scale

        # ============================================================================
        # RETURN
        # ============================================================================
        if return_features:
            return {
                'output': a,
                'encoder': [encoder_feats[0], encoder_feats[1], encoder_feats[2]],
                'decoder': [decoder_feat_144, decoder_feat_36, decoder_feat_9],
            }
        return a



class Generator_320(nn.Module):
    def __init__(self, config, gen_SI=True):
        '''
        Encoder-decoder generator with optional skip connections, producing 320x320 output.
        Designed for domain transformation (e.g., sinogram->image, image->sinogram, or
        iterative reconstruction->ground truth). Supports three skip connection modes:
        none (baseline), add (residual), and concat (U-Net style).

        Architecture: Contracting path (320->160->80->40->20->10), bottleneck neck (narrow/medium/wide -> 1, 5, or 10 spatial size),
        and expanding path (10->20->40->80->160->320). Skip connections cache encoder features at 160, 80, 40, 20, 10
        and optionally merge them into the decoder at matching scales.

        Args:
                config: Dictionary containing network hyperparameters and data dimensions (gen_sino_size, gen_image_size,
                    gen_sino_channels, gen_image_channels). Direction-specific keys (SI_* for sinogram->image,
                    IS_* for image->sinogram) control network architecture:
                    - {SI,IS}_gen_neck: Bottleneck spatial size {'narrow', 'medium', 'wide'} (mapped to 1, 5, 10 for 320)
                    - {SI,IS}_exp_kernel: Expanding kernel size {3, 4}
                    - {SI,IS}_gen_z_dim: Channels in neck for neck=1 (dense bottleneck)
                    - {SI,IS}_gen_hidden_dim: Base channel count; all layers scale by mult**k
                    - {SI,IS}_gen_mult: Multiplicative factor for channels across layers
                    - {SI,IS}_gen_fill: Constant-size conv layers per block {0, 1, 2, 3}
                    - {SI,IS}_layer_norm: Normalization type {'batch', 'instance', 'group', 'none'}
                    - {SI,IS}_pad_mode: Padding mode {'zeros', 'replicate'} (encoder only; decoder uses replicate)
                    - {SI,IS}_dropout: Whether to use dropout {True, False}
                    - {SI,IS}_skip_mode: Skip connection mode {'none', 'add', 'concat'}
                    - {SI,IS}_normalize: Whether to normalize output to L1 norm {True, False}
                    - {SI,IS}_fixedScale: Fixed output scaling factor (ignored if normalize=True)
                    - {SI,IS}_learnedScale_init: Initial value for learnable output scale
                    - {SI,IS}_gen_final_activ: Final activation {nn.Tanh(), nn.Sigmoid(), nn.ReLU(), None}
            gen_SI: Boolean - True for sinogram->image, False for image->sinogram. Controls which
                    config keys (SI_* vs IS_*) are used and input/output dimension assignment.
        '''
        super(Generator_320, self).__init__()

        if gen_SI:
            input_size = config['gen_sino_size']
            input_channels = config['gen_sino_channels']
            output_size = config['gen_image_size']
            output_channels = config['gen_image_channels']

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
            input_size = config['gen_image_size']
            input_channels = config['gen_image_channels']
            output_size = config['gen_sino_size']
            output_channels = config['gen_sino_channels']

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

        self.output_scale_learnable = not bool(config.get(normalize_key, False))
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

        # Root scaling exponent for channel growth
        self.scaling_exp = 0.7  # Change as needed (e.g., 0.7)
        # Unique channel dims for each stage
        dim_0 = int(hidden_dim * mult ** (0 ** self.scaling_exp))
        dim_1 = int(hidden_dim * mult ** (1 ** self.scaling_exp))
        dim_2 = int(hidden_dim * mult ** (2 ** self.scaling_exp))
        dim_3 = int(hidden_dim * mult ** (3 ** self.scaling_exp))
        dim_4 = int(hidden_dim * mult ** (4 ** self.scaling_exp))
        dim_5 = int(hidden_dim * mult ** (5 ** self.scaling_exp))
        dim_6 = int(hidden_dim * mult ** (6 ** self.scaling_exp))

        if input_size != 320:
            raise ValueError('This generator is configured for 320x320 inputs.')

        # Contracting Path: 320 -> 160 -> 80 -> 40 -> 20 -> 10
        self.contract_blocks = nn.ModuleList([
            contract_block(in_chan, dim_0, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),   # 320->160
            contract_block(dim_0, dim_1, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 160->80
            contract_block(dim_1, dim_2, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 80->40
            contract_block(dim_2, dim_3, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 40->20
            contract_block(dim_3, dim_4, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 20->10
        ])

        self.neck = self._build_neck(neck, dim_4, dim_5, dim_6, z_dim, pad, fill, norm, drop)
        self.expand_blocks = self._build_expand(exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, dim_4, pad, fill, norm, drop)

    def _build_neck(self, neck, dim_4, dim_5, dim_6, z_dim, pad, fill, norm, drop):
        # neck='narrow': Narrowest bottleneck (1x1), upsamples back to 10x10 for skip merge
        if neck == 'narrow':
            # ConvTranspose2d formula: H_out = (H_in-1)*stride + kernel - 2*padding + output_padding
            return nn.Sequential(
                contract_block(dim_4, dim_5, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (10+2-4)/2+1 = 5
                contract_block(dim_5, dim_6, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+2-3)/2+1 = 3
                contract_block(dim_6, z_dim, 3, stride=1, padding=0, padding_mode=pad, fill=0, norm='batch', drop=False),     # H = (3+0-3)/1+1 = 1
                expand_block(z_dim, dim_6, 3, stride=2, padding=0, output_padding=0, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # H = (1-1)*2+3-0+0 = 3
                expand_block(dim_6, dim_5, 4, stride=2, padding=2, output_padding=1, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # H = (3-1)*2+4-4+1 = 5
                expand_block(dim_5, dim_4, 3, stride=2, padding=1, output_padding=1, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # H = (5-1)*2+3-2+1 = 10
            )

        # neck='medium': Medium bottleneck (5x5 spatial), constant-size convs; upsamples back to 10x10 for skip merge
        if neck == 'medium':
            return nn.Sequential(
                contract_block(dim_4, dim_5, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (10+2-4)/2+1 = 5
                contract_block(dim_5, dim_5, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                contract_block(dim_5, dim_5, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                contract_block(dim_5, dim_5, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                contract_block(dim_5, dim_5, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                expand_block(dim_5, dim_4, 3, stride=2, padding=1, output_padding=1, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # H = (5-1)*2+3-2+1 = 10
            )

        # neck='wide': Widest bottleneck (10x10 spatial), constant-size layers with kernel=5 for spatial information flow
        if neck == 'wide':
            return nn.Sequential(
                contract_block(dim_4, dim_4, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (10+4-5)/1+1 = 10 (constant)
                contract_block(dim_4, dim_4, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (10+4-5)/1+1 = 10 (constant)
                contract_block(dim_4, dim_4, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (10+4-5)/1+1 = 10 (constant)
                contract_block(dim_4, dim_4, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (10+4-5)/1+1 = 10 (constant)
                contract_block(dim_4, dim_4, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (10+4-5)/1+1 = 10 (constant)
            )

        raise ValueError('neck must be one of {narrow, medium, wide} for Generator')

    def _build_expand(self, exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, dim_4, pad, fill, norm, drop):
        # Expanding Path: 10 -> 20 -> 40 -> 80 -> 160 -> 320
        # ConvTranspose2d formula: H_out = (H_in-1)*stride + kernel - 2*padding + output_padding
        if exp_kernel == 3:
            stage_params = [
                (3, 2, 1, 1),  # 10 -> 20:  H = (10-1)*2+3-2+1 = 20
                (3, 2, 1, 1),  # 20 -> 40:  H = (20-1)*2+3-2+1 = 40
                (3, 2, 1, 1),  # 40 -> 80:  H = (40-1)*2+3-2+1 = 80
                (3, 2, 1, 1),  # 80 -> 160: H = (80-1)*2+3-2+1 = 160
                (3, 2, 1, 1),  # 160 -> 320: H = (160-1)*2+3-2+1 = 320
            ]
        elif exp_kernel == 4:
            stage_params = [
                (4, 2, 1, 0),  # 10 -> 20:  H = (10-1)*2+4-2+0 = 20
                (4, 2, 1, 0),  # 20 -> 40:  H = (20-1)*2+4-2+0 = 40
                (4, 2, 1, 0),  # 40 -> 80:  H = (40-1)*2+4-2+0 = 80
                (4, 2, 1, 0),  # 80 -> 160: H = (80-1)*2+4-2+0 = 160
                (4, 2, 1, 0),  # 160 -> 320: H = (160-1)*2+4-2+0 = 320
            ]
        else:
            raise ValueError('exp_kernel must be 3 or 4')

        def in_ch(base):
            return base * 2 if self.skip_mode == 'concat' else base

        blocks = nn.ModuleList()
        blocks.append(expand_block(in_ch(dim_4), dim_3, stage_params[0][0], stage_params[0][1], stage_params[0][2], stage_params[0][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_3), dim_2, stage_params[1][0], stage_params[1][1], stage_params[1][2], stage_params[1][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_2), dim_1, stage_params[2][0], stage_params[2][1], stage_params[2][2], stage_params[2][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_1), dim_0, stage_params[3][0], stage_params[3][1], stage_params[3][2], stage_params[3][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_0), out_chan, stage_params[4][0], stage_params[4][1], stage_params[4][2], stage_params[4][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop, final_layer=True))
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

        a = self._merge(skips[4], a)
        a = self.expand_blocks[0](a)

        a = self._merge(skips[3], a)
        a = self.expand_blocks[1](a)

        a = self._merge(skips[2], a)
        a = self.expand_blocks[2](a)

        a = self._merge(skips[1], a)
        a = self.expand_blocks[3](a)

        a = self._merge(skips[0], a)
        a = self.expand_blocks[4](a)

        # Center crop to output_size if internal processing was larger
        if a.shape[-1] > self.output_size:
            crop_size = self.output_size
            margin = (a.shape[-1] - crop_size) // 2
            a = a[:, :, margin:margin+crop_size, margin:margin+crop_size]

        if self.final_activation:
            a = self.final_activation(a)
        if self.normalize:
            a = torch.reshape(a, (batch_size, self.output_channels, self.output_size**2))
            a = nn.functional.normalize(a, p=1, dim=2)
            a = torch.reshape(a, (batch_size, self.output_channels, self.output_size, self.output_size))

        scale = torch.exp(self.log_output_scale) if self.output_scale_learnable else self.fixed_output_scale
        a = a * scale
        return a



class Generator_180(nn.Module):
    def __init__(self, config, gen_SI=True):
        '''
        Encoder-decoder generator with optional skip connections, producing 180x180 output.
        Designed for domain transformation (e.g., sinogram->image, image->sinogram, or 
        iterative reconstruction->ground truth). Supports three skip connection modes:
        none (baseline), add (residual), and concat (U-Net style).

        Architecture: Contracting path (180->90->45->23->11), bottleneck neck (narrow/medium/wide -> 1, 5, or 11 spatial size),
        and expanding path (11->23->45->90->180). Skip connections cache encoder features at 90, 45, 23, 11
        and optionally merge them into the decoder at matching scales.

        Args:
                config: Dictionary containing network hyperparameters and data dimensions (gen_sino_size, gen_image_size,
                    gen_sino_channels, gen_image_channels). Direction-specific keys (SI_* for sinogram->image,
                    IS_* for image->sinogram) control network architecture:
                    - {SI,IS}_gen_neck: Bottleneck spatial size {'narrow', 'medium', 'wide'} (mapped to 1, 5, 11 for 180)
                    - {SI,IS}_exp_kernel: Expanding kernel size {3, 4}
                    - {SI,IS}_gen_z_dim: Channels in neck for neck=1 (dense bottleneck)
                    - {SI,IS}_gen_hidden_dim: Base channel count; all layers scale by mult**k
                    - {SI,IS}_gen_mult: Multiplicative factor for channels across layers
                    - {SI,IS}_gen_fill: Constant-size conv layers per block {0, 1, 2, 3}
                    - {SI,IS}_layer_norm: Normalization type {'batch', 'instance', 'group', 'none'}
                    - {SI,IS}_pad_mode: Padding mode {'zeros', 'replicate'} (encoder only; decoder uses replicate)
                    - {SI,IS}_dropout: Whether to use dropout {True, False}
                    - {SI,IS}_skip_mode: Skip connection mode {'none', 'add', 'concat'}
                    - {SI,IS}_normalize: Whether to normalize output to L1 norm {True, False}
                    - {SI,IS}_fixedScale: Fixed output scaling factor (ignored if normalize=True)
                    - {SI,IS}_learnedScale_init: Initial value for learnable output scale
                    - {SI,IS}_gen_final_activ: Final activation {nn.Tanh(), nn.Sigmoid(), nn.ReLU(), None}
            gen_SI: Boolean - True for sinogram->image, False for image->sinogram. Controls which
                    config keys (SI_* vs IS_*) are used and input/output dimension assignment.
        '''
        super(Generator_180, self).__init__()

        if gen_SI:
            input_size = config['gen_sino_size']
            input_channels = config['gen_sino_channels']
            output_size = config['gen_image_size']
            output_channels = config['gen_image_channels']

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
            input_size = config['gen_image_size']
            input_channels = config['gen_image_channels']
            output_size = config['gen_sino_size']
            output_channels = config['gen_sino_channels']

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

        # Root scaling exponent for channel growth
        self.scaling_exp = 0.7
          # Change as needed (e.g., 0.7)
        # Unique channel dims for each stage
        dim_0 = int(hidden_dim * mult ** (0 ** self.scaling_exp))
        dim_1 = int(hidden_dim * mult ** (1 ** self.scaling_exp))
        dim_2 = int(hidden_dim * mult ** (2 ** self.scaling_exp))
        dim_3 = int(hidden_dim * mult ** (3 ** self.scaling_exp))
        dim_4 = int(hidden_dim * mult ** (4 ** self.scaling_exp))
        dim_5 = int(hidden_dim * mult ** (5 ** self.scaling_exp))

        if input_size != 180:
            raise ValueError('This generator is configured for 180x180 inputs and outputs.')

        # Contracting Path: 180 -> 90 -> 45 -> 23 -> 11
        self.contract_blocks = nn.ModuleList([
            contract_block(in_chan, dim_0, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # 180->90
            contract_block(dim_0, dim_1, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),    # 90->45
            contract_block(dim_1, dim_2, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),    # 45->23
            contract_block(dim_2, dim_3, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),    # 23->11
        ])

        self.neck = self._build_neck(neck, dim_3, dim_4, dim_5, z_dim, pad, fill, norm, drop)
        self.expand_blocks = self._build_expand(exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, pad, fill, norm, drop)

    def _build_neck(self, neck, dim_3, dim_4, dim_5, z_dim, pad, fill, norm, drop):
        # neck='narrow': Narrowest bottleneck (1x1), upsamples back to 11x11 for skip merge
        if neck == 'narrow':
            # ConvTranspose2d formula: H_out = (H_in-1)*stride + kernel - 2*padding + output_padding
            return nn.Sequential(
                contract_block(dim_3, dim_4, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (11+2-4)/2+1 = 5
                contract_block(dim_4, dim_5, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+2-3)/2+1 = 3
                contract_block(dim_5, z_dim, 3, stride=1, padding=0, padding_mode=pad, fill=0, norm='batch', drop=False),     # H = (3+0-3)/1+1 = 1
                expand_block(z_dim, dim_5, 3, stride=2, padding=0, output_padding=0, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # H = (1-1)*2+3-0+0 = 3
                expand_block(dim_5, dim_4, 4, stride=2, padding=2, output_padding=1, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # H = (3-1)*2+4-4+1 = 5
                expand_block(dim_4, dim_3, 3, stride=2, padding=0, output_padding=0, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # H = (5-1)*2+3-0+0 = 11
            )

        # neck='medium': Medium bottleneck (5x5 spatial), upsamples back to 11x11 for skip merge
        if neck == 'medium':
            return nn.Sequential(
                contract_block(dim_3, dim_4, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (11+2-4)/2+1 = 5
                contract_block(dim_4, dim_4, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                contract_block(dim_4, dim_4, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                contract_block(dim_4, dim_4, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                contract_block(dim_4, dim_4, 5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),      # H = (5+4-5)/1+1 = 5 (constant)
                expand_block(dim_4, dim_3, 3, stride=2, padding=0, output_padding=0, padding_mode='replicate', fill=fill, norm=norm, drop=drop),  # H = (5-1)*2+3-0+0 = 11
            )

        # neck='wide': Widest bottleneck (11x11 spatial), constant-size layers with kernel=5 for spatial information flow
        if neck == 'wide':
            return nn.Sequential(
                contract_block(dim_3, dim_3, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (11+4-5)/1+1 = 11 (constant)
                contract_block(dim_3, dim_3, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (11+4-5)/1+1 = 11 (constant)
                contract_block(dim_3, dim_3, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (11+4-5)/1+1 = 11 (constant)
                contract_block(dim_3, dim_3, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (11+4-5)/1+1 = 11 (constant)
                contract_block(dim_3, dim_3, kernel_size=5, stride=1, padding=2, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # H = (11+4-5)/1+1 = 11 (constant)
            )

        raise ValueError('neck must be one of {narrow, medium, wide} for Generator_180')

    def _build_expand(self, exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, pad, fill, norm, drop):
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
        blocks.append(expand_block(in_ch(dim_3), dim_2, stage_params[0][0], stage_params[0][1], stage_params[0][2], stage_params[0][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_2), dim_1, stage_params[1][0], stage_params[1][1], stage_params[1][2], stage_params[1][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_1), dim_0, stage_params[2][0], stage_params[2][1], stage_params[2][2], stage_params[2][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_0), out_chan, stage_params[3][0], stage_params[3][1], stage_params[3][2], stage_params[3][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop, final_layer=True))
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

        # Center crop to output_size if internal processing was larger
        if a.shape[-1] > self.output_size:
            crop_size = self.output_size
            margin = (a.shape[-1] - crop_size) // 2
            a = a[:, :, margin:margin+crop_size, margin:margin+crop_size]

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
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode), norm_layer, dropout, nn.ReLU(),
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
        block2 = nn.Sequential(
            norm_layer, dropout, nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode)
            )
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