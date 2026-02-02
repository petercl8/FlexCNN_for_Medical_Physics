import torch
from torch import nn


#############################
##### Generator Classes #####
#############################

class Generator_288(nn.Module):
    def __init__(self, config, gen_SI=True, gen_skip_handling: str = '1x1Conv', gen_flow_mode: str = 'coflow', frozen_enc_channels=None, frozen_dec_channels=None, enable_encoder_mixer=True, enable_decoder_mixer=True, scaling_exp=0.7):
        '''
        Encoder-decoder generator with optional skip connections, producing 288x288 output.
        
        Architecture:
            Contracting path: 288→144→72→36→18→9
            Bottleneck: 1x1, 5x5, or 9x9 spatial size (narrow/medium/wide)
            Expanding path: 9→18→36→72→144→288
        
        Skip Handling Modes:
            - 'classic': Standard U-Net skip connections (add/concat/none). No feature mixing.
            - '1x1Conv': Advanced mode for frozen flow architectures. Skip connections and frozen
                         features are concatenated, then mixed via 1x1 convolutions at decoder stages.
                         Enables transfer learning from frozen backbone networks.
        
        Flow Modes (1x1Conv only):
            - 'coflow': Frozen encoder features → generator encoder stages,
                        Frozen decoder features → generator decoder stages
            - 'counterflow': Frozen features are swapped (encoder↔decoder)
        
        Frozen Feature Dimensions:
            Format: (channels_at_144, channels_at_36, channels_at_9)
            Specifies expected channel counts from frozen backbone at each scale.
            Set to None or (0,0,0) to disable frozen feature mixing at specific network.
        
        Args:
            config: Dictionary with network hyperparameters and data dimensions:
                - gen_sino_size, gen_sino_channels, gen_image_size, gen_image_channels
                - {SI,IS}_gen_neck: 'narrow'/'medium'/'wide'
                - {SI,IS}_exp_kernel: 3 or 4 (expand kernel size)
                - {SI,IS}_gen_z_dim: Channels in narrowest bottleneck
                - {SI,IS}_gen_hidden_dim: Base channel count
                - {SI,IS}_gen_mult: Channel multiplication factor
                - {SI,IS}_gen_fill: Constant-size conv layers per block (0-3)
                - {SI,IS}_layer_norm: 'batch'/'instance'/'group'/'none'
                - {SI,IS}_pad_mode: 'zeros'/'replicate'
                - {SI,IS}_dropout: True/False
                - {SI,IS}_skip_mode: 'none'/'add'/'concat'
                - {SI,IS}_normalize: Normalize output to L1 norm
                - {SI,IS}_fixedScale or {SI,IS}_learnedScale_init: Output scaling
                - {SI,IS}_gen_final_activ: Final activation (nn.Tanh(), etc.)
            gen_SI: True for sinogram→image, False for image→sinogram
            gen_skip_handling: 'classic' or '1x1Conv'
            gen_flow_mode: 'coflow' or 'counterflow' (only used with 1x1Conv)
            frozen_enc_channels: Tuple (ch_144, ch_36, ch_9) for encoder frozen feature mixing
            frozen_dec_channels: Tuple (ch_144, ch_36, ch_9) for decoder frozen feature mixing
            enable_encoder_mixer: Whether to create encoder mixers (default True in 1x1Conv mode)
            enable_decoder_mixer: Whether to create decoder mixers (default True in 1x1Conv mode)
            scaling_exp: Root exponent used to soften channel growth across stages
                         (e.g., channels scale by mult**(k**scaling_exp))
        
        Example Usage:
            # Classic U-Net:
            gen = Generator_288(config, gen_SI=True, gen_skip_handling='classic')
            output = gen(input)
            
            # Frozen flow (receiving features from frozen backbone):
            gen_trainable = Generator_288(config, gen_SI=True,
                                         gen_skip_handling='1x1Conv',
                                         gen_flow_mode='coflow',
                                         frozen_enc_channels=(64, 128, 256),
                                         frozen_dec_channels=(64, 128, 256))
            output = gen_trainable(input, frozen_encoder_features=enc_feats,
                                  frozen_decoder_features=dec_feats)
        '''
        super(Generator_288, self).__init__()

        # ========================================================================
        # PARSE DIRECTION-SPECIFIC CONFIGURATION (SI vs IS)
        # ========================================================================
        direction_config = self._parse_direction_config(config, gen_SI)

        input_size = direction_config['input_size']
        input_channels = direction_config['input_channels']
        output_size = direction_config['output_size']
        output_channels = direction_config['output_channels']
        neck = direction_config['neck']
        exp_kernel = direction_config['exp_kernel']
        z_dim = direction_config['z_dim']
        hidden_dim = direction_config['hidden_dim']
        fill = direction_config['fill']
        mult = direction_config['mult']
        norm = direction_config['norm']
        pad = direction_config['pad']
        drop = direction_config['drop']
        fixed_scale = direction_config['fixed_scale']
        learned_scale_init = direction_config['learned_scale_init']
        
        self.final_activation = direction_config['final_activation']
        self.normalize = direction_config['normalize']
        self.skip_mode = direction_config['skip_mode']

        # ========================================================================
        # FEATURE MIXING CONFIGURATION (1x1Conv mode only)
        # ========================================================================
        self.skip_handling = gen_skip_handling
        if self.skip_handling not in ('classic', '1x1Conv'):
            raise ValueError('gen_skip_handling must be one of {classic, 1x1Conv}')

        self.flow_mode = gen_flow_mode
        if self.flow_mode not in ('coflow', 'counterflow'):
            raise ValueError('gen_flow_mode must be one of {coflow, counterflow}')

        # Normalize frozen feature channel tuples to (ch_144, ch_36, ch_9) format
        self.frozen_enc_channels = self._normalize_injection_tuple(frozen_enc_channels)
        self.frozen_dec_channels = self._normalize_injection_tuple(frozen_dec_channels)
        
        # Validate: mixing requires 1x1Conv mode
        if self.skip_handling == 'classic' and (any(self.frozen_enc_channels) or any(self.frozen_dec_channels)):
            raise ValueError('Frozen feature mixing requires gen_skip_handling="1x1Conv"')
        
        # Store mixer enablement flags (applies only in 1x1Conv mode)
        self.enable_encoder_mixer = enable_encoder_mixer if self.skip_handling == '1x1Conv' else False
        self.enable_decoder_mixer = enable_decoder_mixer if self.skip_handling == '1x1Conv' else False
        
        # ========================================================================
        # OUTPUT SCALING CONFIGURATION
        # ========================================================================
        self.output_scale_learnable = not bool(self.normalize)
        if self.output_scale_learnable:
            init_scale = float(learned_scale_init if learned_scale_init is not None else fixed_scale)
            self.log_output_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        else:
            init_scale = float(fixed_scale)
            self.register_buffer('fixed_output_scale', torch.tensor(init_scale, dtype=torch.float32))

        self.output_channels = output_channels
        self.output_size = output_size

        in_chan = input_channels
        out_chan = output_channels

        # ========================================================================
        # CHANNEL DIMENSION CALCULATION
        # ========================================================================
        # Root scaling with exponent 0.7 produces gentler channel growth than mult**k
        # Example with hidden_dim=16, mult=2.0:
        #   dim_0: 16*2^(0^0.7) = 16*2^0 = 16*1 = 16
        #   dim_1: 16*2^(1^0.7) = 16*2^1 = 16*2 = 32
        #   dim_2: 16*2^(2^0.7) = 16*2^1.62 ≈ 16*3.08 ≈ 49
        #   dim_3: 16*2^(3^0.7) = 16*2^2.16 ≈ 16*4.47 ≈ 72
        # Versus linear mult**k: 16, 32, 64, 128, 256, 512, 1024
        self.scaling_exp = scaling_exp
        dim_0 = int(hidden_dim * mult ** (0 ** self.scaling_exp))
        dim_1 = int(hidden_dim * mult ** (1 ** self.scaling_exp))
        dim_2 = int(hidden_dim * mult ** (2 ** self.scaling_exp))
        dim_3 = int(hidden_dim * mult ** (3 ** self.scaling_exp))
        dim_4 = int(hidden_dim * mult ** (4 ** self.scaling_exp))
        dim_5 = int(hidden_dim * mult ** (5 ** self.scaling_exp))
        dim_6 = int(hidden_dim * mult ** (6 ** self.scaling_exp))

        # Channel references for injection (144,36,9 scales)
        self.enc_stage_channels = (dim_0, dim_2, dim_4)
        # Decoder stages at resolutions 9 (pre first upsample), 36 (pre 36->144 upsample), 144 (pre final upsample)
        self.dec_stage_channels = (dim_0, dim_2, dim_4)
        # Skip connection channels at 144, 72, 36 scales
        self.dec_skip_channels = (dim_0, dim_2, dim_4)

        if input_size != 288:
            raise ValueError('This generator is configured for 288x288 inputs.')

        # ========================================================================
        # BUILD NETWORK ARCHITECTURE
        # ========================================================================
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

        if self.skip_handling == '1x1Conv':
            self._build_mixers()

    # ============================================================================
    # CONFIGURATION HELPERS
    # ============================================================================
    
    def _parse_direction_config(self, config, gen_SI):
        """
        Extract direction-specific configuration (SI vs IS prefixed keys).
        
        Args:
            config: Full configuration dictionary
            gen_SI: True for sinogram→image, False for image→sinogram
        
        Returns:
            Dictionary with normalized keys (without SI_/IS_ prefix)
        """
        if gen_SI:
            return {
                'input_size': config['gen_sino_size'],
                'input_channels': config['gen_sino_channels'],
                'output_size': config['gen_image_size'],
                'output_channels': config['gen_image_channels'],
                'neck': config['SI_gen_neck'],
                'exp_kernel': config['SI_exp_kernel'],
                'z_dim': config['SI_gen_z_dim'],
                'hidden_dim': config['SI_gen_hidden_dim'],
                'fill': config['SI_gen_fill'],
                'mult': config['SI_gen_mult'],
                'norm': config['SI_layer_norm'],
                'pad': config['SI_pad_mode'],
                'drop': config['SI_dropout'],
                'final_activation': config['SI_gen_final_activ'],
                'normalize': config['SI_normalize'],
                'skip_mode': config.get('SI_skip_mode', 'none'),
                'fixed_scale': config.get('SI_fixedScale', 1.0),
                'learned_scale_init': config.get('SI_learnedScale_init'),
            }
        else:
            return {
                'input_size': config['gen_image_size'],
                'input_channels': config['gen_image_channels'],
                'output_size': config['gen_sino_size'],
                'output_channels': config['gen_sino_channels'],
                'neck': config['IS_gen_neck'],
                'exp_kernel': config['IS_exp_kernel'],
                'z_dim': config['IS_gen_z_dim'],
                'hidden_dim': config['IS_gen_hidden_dim'],
                'fill': config['IS_gen_fill'],
                'mult': config['IS_gen_mult'],
                'norm': config['IS_layer_norm'],
                'pad': config['IS_pad_mode'],
                'drop': config['IS_dropout'],
                'final_activation': config['IS_gen_final_activ'],
                'normalize': config['IS_normalize'],
                'skip_mode': config['IS_skip_mode'],
                'fixed_scale': config['IS_fixedScale'],
                'learned_scale_init': config['IS_learnedScale_init'],
            }
    
    def _normalize_injection_tuple(self, cfg):
        """
        Normalize injection channel tuple to (ch_144, ch_36, ch_9) format.
        Ensures that if None is provided, it defaults to (0,0,0).
        
        Args:
            cfg: None, or tuple/list of 3 integers
        
        Returns:
            Tuple of 3 integers (0s if None provided)
        """
        if cfg is None:
            return (0, 0, 0)
        if len(cfg) != 3:
            raise ValueError('Injection tuples must have three entries (144, 36, 9 scales).')
        return tuple(int(x) for x in cfg)

    # ============================================================================
    # ARCHITECTURE BUILDERS
    # ============================================================================
    
    def _build_neck(self, neck, dim_4, dim_5, dim_6, z_dim, pad, fill, norm, drop):
        """
        Build the bottleneck network architecture between encoder and decoder.
        
        Controls information flow at the network's narrowest point. Three modes control
        spatial compression and capacity:
        - 'narrow': 9→5→3→1→3→5→9 (aggressive compression, dense bottleneck)
        - 'medium': 9→5→5→5→5→5→9 (moderate compression, constant 5x5 processing)
        - 'wide': 9→9→9→9→9→9 (no compression, spatial information preserved)
        
        Args:
            neck: Bottleneck mode {'narrow', 'medium', 'wide'}
            dim_4: Channels at encoder output (scale 9)
            dim_5: Channels at intermediate bottleneck stage
            dim_6: Channels at deepest bottleneck stage
            z_dim: Channels at 1x1 spatial bottleneck (narrow mode only)
            pad: Padding mode for convolutions
            fill: Number of constant-size conv layers per block
            norm: Normalization type
            drop: Whether to use dropout
        
        Returns:
            nn.Sequential: Bottleneck network module
        """
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
        """
        Build the decoder (expanding path) from bottleneck to output resolution.
        
        Creates 5 upsampling stages: 9→18→36→72→144→288. Each stage uses transposed
        convolutions with optional fill layers. Input channels are adjusted based on
        skip connection mode (doubled for concat mode in classic skip handling).
        
        Args:
            exp_kernel: Kernel size for transposed convolutions {3, 4}
            out_chan: Output channels (final image/sinogram channels)
            dim_0: Channels at scale 144
            dim_1: Channels at scale 72
            dim_2: Channels at scale 36
            dim_3: Channels at scale 18
            dim_4: Channels at scale 9
            pad: Padding mode (decoder uses 'replicate')
            fill: Number of constant-size conv layers per block
            norm: Normalization type
            drop: Whether to use dropout
            skip_handling: Skip connection mode {'classic', '1x1Conv'}
        
        Returns:
            nn.ModuleList: List of 5 expand blocks for upsampling stages
        """
        # Expanding Path: 9 -> 18 -> 36 -> 72 -> 144 -> 288
        if exp_kernel == 3:
            stage_params = [   # kernel, stride, padding, output_padding
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
            """
            Calculate input channels for expand block.
            In classic mode with concat: doubles channels to accommodate skip connection
            In 1x1Conv mode: base channels (merging handled by injectors)
            """
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

    def _build_mixers(self):
        """
        Build 1x1 convolutional projection layers for feature mixing in 1x1Conv mode.
        
        Creates two sets of mixers:
        - Encoder mixers: Mix frozen encoder features at scales 144, 36, 9 (if enabled)
        - Decoder mixers: Mix skip connections + frozen decoder features (always enabled in 1x1Conv)
        
        Each mixer concatenates multiple feature sources (base features + optional skip + optional frozen),
        then projects back to base channel count via 1x1 conv. Always created when enabled.
                
        Returns:
            None (creates self.enc_mixers and self.dec_mixers ModuleDicts)
        """
        def _make_proj(in_ch, out_ch):
            return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        # Encoder mixers (created if enabled)
        enc_keys = ['enc_144', 'enc_36', 'enc_9']
        enc_chs = self.enc_stage_channels
        self.enc_mixers = nn.ModuleDict()
        if self.enable_encoder_mixer:
            for key, base_ch, frozen_ch in zip(enc_keys, enc_chs, self.frozen_enc_channels):
                # Mixer handles base channels + any frozen channels at this scale
                self.enc_mixers[key] = _make_proj(base_ch + frozen_ch, base_ch)

        # Decoder mixers (always created in 1x1Conv mode)
        dec_keys = ['dec_144', 'dec_36', 'dec_9']
        dec_chs = self.dec_stage_channels
        skip_chs = self.dec_skip_channels
        self.dec_mixers = nn.ModuleDict()
        for key, base_ch, skip_ch, frozen_ch in zip(dec_keys, dec_chs, skip_chs, self.frozen_dec_channels):
            total_in = base_ch + frozen_ch
            if self.skip_mode != 'none':
                total_in += skip_ch
            self.dec_mixers[key] = _make_proj(total_in, base_ch)

    # ============================================================================
    # FORWARD PASS HELPERS
    # ============================================================================
    
    def _validate_and_route_frozen_features(self, frozen_encoder_features, frozen_decoder_features):
        """
        Validate and route frozen features based on skip_handling and flow_mode.
        
        Args:
            frozen_encoder_features: Encoder features from frozen network (or None)
            frozen_decoder_features: Decoder features from frozen network (or None)
        
        Returns:
            (routed_encoder_features, routed_decoder_features) as tuples or None
        """

        # Route features based on flow mode
        routed_enc_features = frozen_encoder_features
        routed_dec_features = frozen_decoder_features
        if self.flow_mode == 'counterflow':
            # Counterflow: swap encoder and decoder features
            routed_enc_features, routed_dec_features = routed_dec_features, routed_enc_features

        # Convert to tuples for indexing
        routed_enc_features = tuple(routed_enc_features) if routed_enc_features is not None else None
        routed_dec_features = tuple(routed_dec_features) if routed_dec_features is not None else None

        # Validate: frozen encoder features require encoder mixer to be enabled
        if routed_enc_features is not None and not self.enable_encoder_mixer:
            raise ValueError('Frozen encoder features provided but enable_encoder_mixer=False')
        
        # Validate: frozen decoder features require decoder mixer to be enabled
        if routed_dec_features is not None and not self.enable_decoder_mixer:
            raise ValueError('Frozen decoder features provided but enable_decoder_mixer=False')

        # Validate: classic mode cannot accept frozen features
        if self.skip_handling == 'classic' and (frozen_encoder_features is not None or frozen_decoder_features is not None):
            raise ValueError('Frozen features provided but gen_skip_handling is classic.')

        return routed_enc_features, routed_dec_features
    
    def _validate_feature_shape(self, tensor, target_h, target_w, expected_c, label):
        """
        Validate that a feature tensor has expected spatial size and channel count.
        
        Args:
            tensor: Feature tensor to validate
            target_h: Expected height
            target_w: Expected width
            expected_c: Expected channel count
            label: Descriptive label for error messages
        
        Raises:
            ValueError: If shape doesn't match expectations
        """
        if tensor is None:
            raise ValueError(f'Missing tensor for {label}')
        h, w = tensor.shape[-2:]
        if h != target_h or w != target_w:
            raise ValueError(f'Shape mismatch for {label}: expected {target_h}x{target_w}, got {h}x{w}')
        if tensor.shape[1] != expected_c:
            raise ValueError(f'Channel mismatch for {label}: expected {expected_c}, got {tensor.shape[1]}')
    
    def _inject_and_merge_at_decoder_stage(self, hidden, skip_tensor, frozen_feature,
                                           inject_idx, mixer_key, return_features):
        """
        Mix skip connection and frozen decoder features at a decoder stage.
        
        Handles both classic mode (simple merge) and 1x1Conv mode (concat + projection).
        
        Args:
            hidden: Current decoder output tensor
            skip_tensor: Skip connection from encoder (always passed, even if not used)
            frozen_feature: Frozen feature to mix (or None)
            inject_idx: Index into frozen_dec_channels (0, 1, or 2)
            mixer_key: Key for dec_mixers ModuleDict
            return_features: Whether to capture features for return
        
        Returns:
            (updated_hidden, optional_captured_feature)
        """
        captured_feature = None
        
        if self.skip_handling == '1x1Conv':
            # 1x1Conv mode: concatenate decoder output + optional skip + optional frozen feature
            parts = [hidden]
            if self.skip_mode != 'none':
                parts.append(skip_tensor)

            if self.frozen_dec_channels[inject_idx] > 0:
                frozen_channels_expected = self.frozen_dec_channels[inject_idx]
                self._validate_feature_shape(frozen_feature, skip_tensor.shape[-2],
                                            skip_tensor.shape[-1], frozen_channels_expected,
                                            f'decoder_mix_{mixer_key}')
                parts.append(frozen_feature)

            # Concatenate and project back to base channels
            hidden = torch.cat(parts, dim=1)
            hidden = self.dec_mixers[mixer_key](hidden)

            # Capture features AFTER mixing
            if return_features:
                captured_feature = hidden
        else:
            # Classic mode: simple skip connection merge
            if return_features:
                captured_feature = hidden # Capture BEFORE merging because otherwise the channel number would be 2X larger
            hidden = self._merge_classic(skip_tensor, hidden)
        
        return hidden, captured_feature
    
    def _merge_classic(self, skip, x):
        """
        Merge skip connection with decoder output using configured skip_mode.
        Used in classic mode only; 1x1Conv mode uses _inject_and_merge_at_decoder_stage.
        
        Args:
            skip: Skip connection tensor from encoder
            x: Current decoder output tensor
        
        Returns:
            Merged tensor (add, concat, or passthrough if skip_mode='none')
        """
        if self.skip_mode == 'none' or skip is None:
            return x
        if self.skip_mode == 'add':
            return x + skip
        if self.skip_mode == 'concat':
            return torch.cat([x, skip], dim=1)
        raise ValueError('skip_mode must be one of {none, add, concat}')

    def forward(self, input, frozen_encoder_features=None, frozen_decoder_features=None, return_features: bool = False):
        batch_size = len(input)

        # ================================================================================
        # SECTION 1: SETUP AND VALIDATION
        # ================================================================================
        routed_enc_features, routed_dec_features = self._validate_and_route_frozen_features(
            frozen_encoder_features, frozen_decoder_features
        ) # routed_enc_features: tuple or None; routed_dec_features: tuple or None

        # ================================================================================
        # SECTION 2: ENCODER (Contracting Path: 288 → 144 → 72 → 36 → 18 → 9)
        # ================================================================================
        # Inject frozen features at three spatial scales during encoder:
        # Scale 144 (idx=0), Scale 36 (idx=2), Scale 9 (idx=4)
        # Pattern: Concatenate frozen features → 1x1 conv projects back to base channels
        skips = []
        hidden = input
        
        for idx, block in enumerate(self.contract_blocks):
            hidden = block(hidden)
            
            # Mix frozen encoder features at scales 144, 36, 9
            if self.enable_encoder_mixer and idx in (0, 2, 4):
                inj_idx = {0: 0, 2: 1, 4: 2}[idx]
                key = ('enc_144', 'enc_36', 'enc_9')[inj_idx]
                
                # If frozen channels are configured for this scale, validate and concatenate
                if self.frozen_enc_channels[inj_idx] > 0:
                    frozen_feature = routed_enc_features[inj_idx]
                    frozen_channels_expected = self.frozen_enc_channels[inj_idx]
                    self._validate_feature_shape(frozen_feature, hidden.shape[-2], hidden.shape[-1],
                                                frozen_channels_expected, f'encoder_mix_{inj_idx}')
                    hidden = torch.cat([hidden, frozen_feature], dim=1)  # Concatenate along channel dimension
                
                # Always apply mixer (acts as channel mixer even when no frozen features)
                hidden = self.enc_mixers[key](hidden)
            
            skips.append(hidden) # Skips are appended for every layer because in 'classic' mode we need all of them

        if return_features:
            encoder_feats = [skips[0], skips[2], skips[4]]  # Scales: 144, 36, 9

        # ================================================================================
        # SECTION 3: BOTTLENECK (Processing at smallest spatial scale)
        # ================================================================================
        hidden = self.neck(hidden)

        # ================================================================================
        # SECTION 4: DECODER (Expanding Path: 9 → 18 → 36 → 72 → 144 → 288)
        # ================================================================================
        # Note: In '1x1Conv' mode, frozen features are injected at scales 9, 72, 144
        #       In 'classic' mode, skip connections merged at scales 9, 18, 36, 72, 144

        # --- Stage 1: Mix/merge at scale 9, then upsample to 18 ---
        routed_dec_feature9 = routed_dec_features[2] if routed_dec_features is not None else None
        hidden, decoder_feat_scale9 = self._inject_and_merge_at_decoder_stage(
            hidden, skips[4], routed_dec_feature9, inject_idx=2, mixer_key='dec_9', 
            return_features=return_features
        )
        hidden = self.expand_blocks[0](hidden)  # 9 → 18

        # --- Stage 2: Merge skip at 18x18 (classic only), then upsample to 36 ---
        if self.skip_handling == 'classic':
            hidden = self._merge_classic(skips[3], hidden)
        hidden = self.expand_blocks[1](hidden)  # 18 → 36

        # --- Stage 3: Mix/merge at scale 36, then upsample to 72 ---
        if self.skip_handling == '1x1Conv':
            routed_dec_feature36 = routed_dec_features[1] if routed_dec_features is not None else None
            hidden, decoder_feat_scale36 = self._inject_and_merge_at_decoder_stage(
                hidden, skips[2], routed_dec_feature36, inject_idx=1, mixer_key='dec_36',
                return_features=return_features
            )
        else:
            # Classic: merge skip before expand
            hidden = self._merge_classic(skips[2], hidden)
            if return_features:
                decoder_feat_scale36 = hidden
        hidden = self.expand_blocks[2](hidden)  # 36 → 72

        # --- Stage 4: Upsample 72 → 144 (skip handled by classic mode only) ---
        if self.skip_handling == 'classic':
            hidden = self._merge_classic(skips[1], hidden)
        hidden = self.expand_blocks[3](hidden)  # 72 → 144

        # --- Stage 5: Mix/merge at scale 144, then upsample to 288 ---
        
        # Note: a 'classic' mode skip connection is not employed here because you don't want raw skips at final output scale
        routed_dec_feature144 = routed_dec_features[0] if routed_dec_features is not None else None
        hidden, decoder_feat_scale144 = self._inject_and_merge_at_decoder_stage(
            hidden, skips[0], routed_dec_feature144, inject_idx=0, mixer_key='dec_144',
            return_features=return_features
        )
        hidden = self.expand_blocks[4](hidden)  # 144 → 288

        # ================================================================================
        # SECTION 5: POST-PROCESSING (Activation, Normalization, Scaling)
        # ================================================================================
        # Center crop if output exceeds target size
        if hidden.shape[-1] > self.output_size:
            crop_size = self.output_size
            margin = (hidden.shape[-1] - crop_size) // 2
            hidden = hidden[:, :, margin:margin+crop_size, margin:margin+crop_size]

        # Apply final activation (Tanh, Sigmoid, etc.)
        if self.final_activation:
            hidden = self.final_activation(hidden)

        # L1 normalization across spatial dimensions (if enabled)
        if self.normalize:
            hidden = torch.reshape(hidden, (batch_size, self.output_channels, self.output_size**2))
            hidden = nn.functional.normalize(hidden, p=1, dim=2)
            hidden = torch.reshape(hidden, (batch_size, self.output_channels, self.output_size, self.output_size))

        # Apply output scaling (learnable or fixed)
        scale = torch.exp(self.log_output_scale) if self.output_scale_learnable else self.fixed_output_scale
        output = hidden * scale

        # ================================================================================
        # SECTION 6: RETURN OUTPUT (with optional intermediate features)
        # ================================================================================
        if return_features:
            return {
                'output': output,
                'encoder': [encoder_feats[0], encoder_feats[1], encoder_feats[2]],
                'decoder': [decoder_feat_scale144, decoder_feat_scale36, decoder_feat_scale9],
            }
        return output



class Generator_320(nn.Module):
    def __init__(self, config, gen_SI=True, gen_skip_handling: str = '1x1Conv', gen_flow_mode: str = 'coflow', frozen_enc_channels=None, frozen_dec_channels=None, enable_encoder_mixer=True, enable_decoder_mixer=True, scaling_exp=0.7):
        '''
        Encoder-decoder generator with optional skip connections, producing 320x320 output.
        
        Architecture:
            Contracting path: 320→160→80→40→20→10
            Bottleneck: 1x1, 5x5, or 10x10 spatial size (narrow/medium/wide)
            Expanding path: 10→20→40→80→160→320
        
        Skip Handling Modes:
            - 'classic': Standard U-Net skip connections (add/concat/none). No feature mixing.
            - '1x1Conv': Advanced mode for frozen flow architectures. Skip connections and frozen
                         features are concatenated, then mixed via 1x1 convolutions at decoder stages.
                         Enables transfer learning from frozen backbone networks.
        
        Flow Modes (1x1Conv only):
            - 'coflow': Frozen encoder features → generator encoder stages,
                        Frozen decoder features → generator decoder stages
            - 'counterflow': Frozen features are swapped (encoder↔decoder)
        
        Frozen Feature Dimensions:
            Format: (channels_at_160, channels_at_40, channels_at_10)
            Specifies expected channel counts from frozen backbone at each scale.
            Set to None or (0,0,0) to disable frozen feature mixing at specific network.
        
        Args:
            config: Dictionary with network hyperparameters and data dimensions:
                - gen_sino_size, gen_sino_channels, gen_image_size, gen_image_channels
                - {SI,IS}_gen_neck: 'narrow'/'medium'/'wide'
                - {SI,IS}_exp_kernel: 3 or 4 (expand kernel size)
                - {SI,IS}_gen_z_dim: Channels in narrowest bottleneck
                - {SI,IS}_gen_hidden_dim: Base channel count
                - {SI,IS}_gen_mult: Channel multiplication factor
                - {SI,IS}_gen_fill: Constant-size conv layers per block (0-3)
                - {SI,IS}_layer_norm: 'batch'/'instance'/'group'/'none'
                - {SI,IS}_pad_mode: 'zeros'/'replicate'
                - {SI,IS}_dropout: True/False
                - {SI,IS}_skip_mode: 'none'/'add'/'concat'
                - {SI,IS}_normalize: Normalize output to L1 norm
                - {SI,IS}_fixedScale or {SI,IS}_learnedScale_init: Output scaling
                - {SI,IS}_gen_final_activ: Final activation (nn.Tanh(), etc.)
            gen_SI: True for sinogram→image, False for image→sinogram
            gen_skip_handling: 'classic' or '1x1Conv'
            gen_flow_mode: 'coflow' or 'counterflow' (only used with 1x1Conv)
            frozen_enc_channels: Tuple (ch_160, ch_40, ch_10) for encoder frozen feature mixing
            frozen_dec_channels: Tuple (ch_160, ch_40, ch_10) for decoder frozen feature mixing
            enable_encoder_mixer: Whether to create encoder mixers (default True in 1x1Conv mode)
            enable_decoder_mixer: Whether to create decoder mixers (default True in 1x1Conv mode)
            scaling_exp: Root exponent used to soften channel growth across stages
                         (e.g., channels scale by mult**(k**scaling_exp))
        
        Example Usage:
            # Classic U-Net:
            gen = Generator_320(config, gen_SI=True, gen_skip_handling='classic')
            output = gen(input)
            
            # Frozen flow (receiving features from frozen backbone):
            gen_trainable = Generator_320(config, gen_SI=True,
                                         gen_skip_handling='1x1Conv',
                                         gen_flow_mode='coflow',
                                         frozen_enc_channels=(64, 128, 256),
                                         frozen_dec_channels=(64, 128, 256))
            output = gen_trainable(input, frozen_encoder_features=enc_feats,
                                  frozen_decoder_features=dec_feats)
        '''
        super(Generator_320, self).__init__()

        # ========================================================================
        # PARSE DIRECTION-SPECIFIC CONFIGURATION (SI vs IS)
        # ========================================================================
        direction_config = self._parse_direction_config(config, gen_SI)

        input_size = direction_config['input_size']
        input_channels = direction_config['input_channels']
        output_size = direction_config['output_size']
        output_channels = direction_config['output_channels']
        neck = direction_config['neck']
        exp_kernel = direction_config['exp_kernel']
        z_dim = direction_config['z_dim']
        hidden_dim = direction_config['hidden_dim']
        fill = direction_config['fill']
        mult = direction_config['mult']
        norm = direction_config['norm']
        pad = direction_config['pad']
        drop = direction_config['drop']
        fixed_scale = direction_config['fixed_scale']
        learned_scale_init = direction_config['learned_scale_init']
        
        self.final_activation = direction_config['final_activation']
        self.normalize = direction_config['normalize']
        self.skip_mode = direction_config['skip_mode']

        # ========================================================================
        # FEATURE MIXING CONFIGURATION (1x1Conv mode only)
        # ========================================================================
        self.skip_handling = gen_skip_handling
        if self.skip_handling not in ('classic', '1x1Conv'):
            raise ValueError('gen_skip_handling must be one of {classic, 1x1Conv}')

        self.flow_mode = gen_flow_mode
        if self.flow_mode not in ('coflow', 'counterflow'):
            raise ValueError('gen_flow_mode must be one of {coflow, counterflow}')

        # Normalize frozen feature channel tuples to (ch_160, ch_40, ch_10) format
        self.frozen_enc_channels = self._normalize_injection_tuple(frozen_enc_channels)
        self.frozen_dec_channels = self._normalize_injection_tuple(frozen_dec_channels)
        
        # Validate: mixing requires 1x1Conv mode
        if self.skip_handling == 'classic' and (any(self.frozen_enc_channels) or any(self.frozen_dec_channels)):
            raise ValueError('Frozen feature mixing requires gen_skip_handling="1x1Conv"')
        
        # Store mixer enablement flags (applies only in 1x1Conv mode)
        self.enable_encoder_mixer = enable_encoder_mixer if self.skip_handling == '1x1Conv' else False
        self.enable_decoder_mixer = enable_decoder_mixer if self.skip_handling == '1x1Conv' else False
        
        # ========================================================================
        # OUTPUT SCALING CONFIGURATION
        # ========================================================================
        self.output_scale_learnable = not bool(self.normalize)
        if self.output_scale_learnable:
            init_scale = float(learned_scale_init if learned_scale_init is not None else fixed_scale)
            self.log_output_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        else:
            init_scale = float(fixed_scale)
            self.register_buffer('fixed_output_scale', torch.tensor(init_scale, dtype=torch.float32))

        self.output_channels = output_channels
        self.output_size = output_size

        in_chan = input_channels
        out_chan = output_channels

        # ========================================================================
        # CHANNEL DIMENSION CALCULATION
        # ========================================================================
        # Root scaling with exponent 0.7 produces gentler channel growth than mult**k
        self.scaling_exp = scaling_exp
        dim_0 = int(hidden_dim * mult ** (0 ** self.scaling_exp))
        dim_1 = int(hidden_dim * mult ** (1 ** self.scaling_exp))
        dim_2 = int(hidden_dim * mult ** (2 ** self.scaling_exp))
        dim_3 = int(hidden_dim * mult ** (3 ** self.scaling_exp))
        dim_4 = int(hidden_dim * mult ** (4 ** self.scaling_exp))
        dim_5 = int(hidden_dim * mult ** (5 ** self.scaling_exp))
        dim_6 = int(hidden_dim * mult ** (6 ** self.scaling_exp))

        # Channel references for mixing (160, 40, 10 scales)
        self.enc_stage_channels = (dim_0, dim_2, dim_4)
        # Decoder stages at resolutions 10 (pre first upsample), 40 (pre 40->160 upsample), 160 (pre final upsample)
        self.dec_stage_channels = (dim_0, dim_2, dim_4)
        # Skip connection channels at 160, 40, 10 scales
        self.dec_skip_channels = (dim_0, dim_2, dim_4)

        if input_size != 320:
            raise ValueError('This generator is configured for 320x320 inputs.')

        # ========================================================================
        # BUILD NETWORK ARCHITECTURE
        # ========================================================================
        # Contracting Path: 320 -> 160 -> 80 -> 40 -> 20 -> 10
        self.contract_blocks = nn.ModuleList([
            contract_block(in_chan, dim_0, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),   # 320->160
            contract_block(dim_0, dim_1, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 160->80
            contract_block(dim_1, dim_2, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 80->40
            contract_block(dim_2, dim_3, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 40->20
            contract_block(dim_3, dim_4, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),     # 20->10
        ])

        self.neck = self._build_neck(neck, dim_4, dim_5, dim_6, z_dim, pad, fill, norm, drop)
        self.expand_blocks = self._build_expand(exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, dim_4, pad, fill, norm, drop, self.skip_handling)

        if self.skip_handling == '1x1Conv':
            self._build_mixers()

    # ============================================================================
    # CONFIGURATION HELPERS
    # ============================================================================
    
    def _parse_direction_config(self, config, gen_SI):
        """
        Extract direction-specific configuration (SI vs IS prefixed keys).
        
        Args:
            config: Full configuration dictionary
            gen_SI: True for sinogram→image, False for image→sinogram
        
        Returns:
            Dictionary with normalized keys (without SI_/IS_ prefix)
        """
        if gen_SI:
            return {
                'input_size': config['gen_sino_size'],
                'input_channels': config['gen_sino_channels'],
                'output_size': config['gen_image_size'],
                'output_channels': config['gen_image_channels'],
                'neck': config['SI_gen_neck'],
                'exp_kernel': config['SI_exp_kernel'],
                'z_dim': config['SI_gen_z_dim'],
                'hidden_dim': config['SI_gen_hidden_dim'],
                'fill': config['SI_gen_fill'],
                'mult': config['SI_gen_mult'],
                'norm': config['SI_layer_norm'],
                'pad': config['SI_pad_mode'],
                'drop': config['SI_dropout'],
                'final_activation': config['SI_gen_final_activ'],
                'normalize': config['SI_normalize'],
                'skip_mode': config.get('SI_skip_mode', 'none'),
                'fixed_scale': config.get('SI_fixedScale', 1.0),
                'learned_scale_init': config.get('SI_learnedScale_init'),
            }
        else:
            return {
                'input_size': config['gen_image_size'],
                'input_channels': config['gen_image_channels'],
                'output_size': config['gen_sino_size'],
                'output_channels': config['gen_sino_channels'],
                'neck': config['IS_gen_neck'],
                'exp_kernel': config['IS_exp_kernel'],
                'z_dim': config['IS_gen_z_dim'],
                'hidden_dim': config['IS_gen_hidden_dim'],
                'fill': config['IS_gen_fill'],
                'mult': config['IS_gen_mult'],
                'norm': config['IS_layer_norm'],
                'pad': config['IS_pad_mode'],
                'drop': config['IS_dropout'],
                'final_activation': config['IS_gen_final_activ'],
                'normalize': config['IS_normalize'],
                'skip_mode': config['IS_skip_mode'],
                'fixed_scale': config['IS_fixedScale'],
                'learned_scale_init': config['IS_learnedScale_init'],
            }
    
    def _normalize_injection_tuple(self, cfg):
        """
        Normalize injection channel tuple to (ch_160, ch_40, ch_10) format.
        Ensures that if None is provided, it defaults to (0,0,0).
        
        Args:
            cfg: None, or tuple/list of 3 integers
        
        Returns:
            Tuple of 3 integers (0s if None provided)
        """
        if cfg is None:
            return (0, 0, 0)
        if len(cfg) != 3:
            raise ValueError('Injection tuples must have three entries (160, 40, 10 scales).')
        return tuple(int(x) for x in cfg)

    # ============================================================================
    # ARCHITECTURE BUILDERS
    # ============================================================================

    def _build_neck(self, neck, dim_4, dim_5, dim_6, z_dim, pad, fill, norm, drop):
        """
        Build the bottleneck network architecture between encoder and decoder.
        
        Controls information flow at the network's narrowest point. Three modes control
        spatial compression and capacity:
        - 'narrow': 10→5→3→1→3→5→10 (aggressive compression, dense bottleneck)
        - 'medium': 10→5→5→5→5→5→10 (moderate compression, constant 5x5 processing)
        - 'wide': 10→10→10→10→10→10 (no compression, spatial information preserved)
        
        Args:
            neck: Bottleneck mode {'narrow', 'medium', 'wide'}
            dim_4: Channels at encoder output (scale 10)
            dim_5: Channels at intermediate bottleneck stage
            dim_6: Channels at deepest bottleneck stage
            z_dim: Channels at 1x1 spatial bottleneck (narrow mode only)
            pad: Padding mode for convolutions
            fill: Number of constant-size conv layers per block
            norm: Normalization type
            drop: Whether to use dropout
        
        Returns:
            nn.Sequential: Bottleneck network module
        """
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

    def _build_expand(self, exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, dim_4, pad, fill, norm, drop, skip_handling):
        """
        Build the decoder (expanding path) from bottleneck to output resolution.
        
        Creates 5 upsampling stages: 10→20→40→80→160→320. Each stage uses transposed
        convolutions with optional fill layers. Input channels are adjusted based on
        skip connection mode (doubled for concat mode in classic skip handling).
        
        Args:
            exp_kernel: Kernel size for transposed convolutions {3, 4}
            out_chan: Output channels (final image/sinogram channels)
            dim_0: Channels at scale 160
            dim_1: Channels at scale 80
            dim_2: Channels at scale 40
            dim_3: Channels at scale 20
            dim_4: Channels at scale 10
            pad: Padding mode (decoder uses 'replicate')
            fill: Number of constant-size conv layers per block
            norm: Normalization type
            drop: Whether to use dropout
            skip_handling: Skip connection mode {'classic', '1x1Conv'}
        
        Returns:
            nn.ModuleList: List of 5 expand blocks for upsampling stages
        """
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
            """
            Calculate input channels for expand block.
            In classic mode with concat: doubles channels to accommodate skip connection
            In 1x1Conv mode: base channels (merging handled by mixers)
            """
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

    def _build_mixers(self):
        """
        Build 1x1 convolutional projection layers for feature mixing in 1x1Conv mode.
        
        Creates two sets of mixers:
        - Encoder mixers: Mix frozen encoder features at scales 160, 40, 10 (if enabled)
        - Decoder mixers: Mix skip connections + frozen decoder features (always enabled in 1x1Conv)
        
        Each mixer concatenates multiple feature sources (base features + optional skip + optional frozen),
        then projects back to base channel count via 1x1 conv. Always created when enabled.
                
        Returns:
            None (creates self.enc_mixers and self.dec_mixers ModuleDicts)
        """
        def _make_proj(in_ch, out_ch):
            return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        # Encoder mixers (created if enabled)
        enc_keys = ['enc_160', 'enc_40', 'enc_10']
        enc_chs = self.enc_stage_channels
        self.enc_mixers = nn.ModuleDict()
        if self.enable_encoder_mixer:
            for key, base_ch, frozen_ch in zip(enc_keys, enc_chs, self.frozen_enc_channels):
                # Mixer handles base channels + any frozen channels at this scale
                self.enc_mixers[key] = _make_proj(base_ch + frozen_ch, base_ch)

        # Decoder mixers (always created in 1x1Conv mode)
        dec_keys = ['dec_160', 'dec_40', 'dec_10']
        dec_chs = self.dec_stage_channels
        skip_chs = self.dec_skip_channels
        self.dec_mixers = nn.ModuleDict()
        for key, base_ch, skip_ch, frozen_ch in zip(dec_keys, dec_chs, skip_chs, self.frozen_dec_channels):
            total_in = base_ch + frozen_ch
            if self.skip_mode != 'none':
                total_in += skip_ch
            self.dec_mixers[key] = _make_proj(total_in, base_ch)

    # ============================================================================
    # FORWARD PASS HELPERS
    # ============================================================================
    
    def _validate_and_route_frozen_features(self, frozen_encoder_features, frozen_decoder_features):
        """
        Validate and route frozen features based on skip_handling and flow_mode.
        
        Args:
            frozen_encoder_features: Encoder features from frozen network (or None)
            frozen_decoder_features: Decoder features from frozen network (or None)
        
        Returns:
            (routed_encoder_features, routed_decoder_features) as tuples or None
        """

        # Route features based on flow mode
        routed_enc_features = frozen_encoder_features
        routed_dec_features = frozen_decoder_features
        if self.flow_mode == 'counterflow':
            # Counterflow: swap encoder and decoder features
            routed_enc_features, routed_dec_features = routed_dec_features, routed_enc_features

        # Convert to tuples for indexing
        routed_enc_features = tuple(routed_enc_features) if routed_enc_features is not None else None
        routed_dec_features = tuple(routed_dec_features) if routed_dec_features is not None else None

        # Validate: frozen encoder features require encoder mixer to be enabled
        if routed_enc_features is not None and not self.enable_encoder_mixer:
            raise ValueError('Frozen encoder features provided but enable_encoder_mixer=False')
        
        # Validate: frozen decoder features require decoder mixer to be enabled
        if routed_dec_features is not None and not self.enable_decoder_mixer:
            raise ValueError('Frozen decoder features provided but enable_decoder_mixer=False')

        # Validate: classic mode cannot accept frozen features
        if self.skip_handling == 'classic' and (frozen_encoder_features is not None or frozen_decoder_features is not None):
            raise ValueError('Frozen features provided but gen_skip_handling is classic.')

        return routed_enc_features, routed_dec_features
    
    def _validate_feature_shape(self, tensor, target_h, target_w, expected_c, label):
        """
        Validate that a feature tensor has expected spatial size and channel count.
        
        Args:
            tensor: Feature tensor to validate
            target_h: Expected height
            target_w: Expected width
            expected_c: Expected channel count
            label: Descriptive label for error messages
        
        Raises:
            ValueError: If shape doesn't match expectations
        """
        if tensor is None:
            raise ValueError(f'Missing tensor for {label}')
        h, w = tensor.shape[-2:]
        if h != target_h or w != target_w:
            raise ValueError(f'Shape mismatch for {label}: expected {target_h}x{target_w}, got {h}x{w}')
        if tensor.shape[1] != expected_c:
            raise ValueError(f'Channel mismatch for {label}: expected {expected_c}, got {tensor.shape[1]}')
    
    def _inject_and_merge_at_decoder_stage(self, hidden, skip_tensor, frozen_feature,
                                           inject_idx, mixer_key, return_features):
        """
        Mix skip connection and frozen decoder features at a decoder stage.
        
        Handles both classic mode (simple merge) and 1x1Conv mode (concat + projection).
        
        Args:
            hidden: Current decoder output tensor
            skip_tensor: Skip connection from encoder (always passed, even if not used)
            frozen_feature: Frozen feature to mix (or None)
            inject_idx: Index into frozen_dec_channels (0, 1, or 2)
            mixer_key: Key for dec_mixers ModuleDict
            return_features: Whether to capture features for return
        
        Returns:
            (updated_hidden, optional_captured_feature)
        """
        captured_feature = None
        
        if self.skip_handling == '1x1Conv':
            # 1x1Conv mode: concatenate decoder output + optional skip + optional frozen feature
            parts = [hidden]
            if self.skip_mode != 'none':
                parts.append(skip_tensor)

            if self.frozen_dec_channels[inject_idx] > 0:
                frozen_channels_expected = self.frozen_dec_channels[inject_idx]
                self._validate_feature_shape(frozen_feature, skip_tensor.shape[-2],
                                            skip_tensor.shape[-1], frozen_channels_expected,
                                            f'decoder_mix_{mixer_key}')
                parts.append(frozen_feature)

            # Concatenate and project back to base channels
            hidden = torch.cat(parts, dim=1)
            hidden = self.dec_mixers[mixer_key](hidden)

            # Capture features AFTER mixing
            if return_features:
                captured_feature = hidden
        else:
            # Classic mode: simple skip connection merge
            if return_features:
                captured_feature = hidden # Capture BEFORE merging because otherwise the channel number would be 2X larger
            hidden = self._merge_classic(skip_tensor, hidden)
        
        return hidden, captured_feature

    def _merge_classic(self, skip, x):
        """
        Merge skip connection with decoder output using configured skip_mode.
        Used in classic mode only; 1x1Conv mode uses _inject_and_merge_at_decoder_stage.
        
        Args:
            skip: Skip connection tensor from encoder
            x: Current decoder output tensor
        
        Returns:
            Merged tensor (add, concat, or passthrough if skip_mode='none')
        """
        if self.skip_mode == 'none' or skip is None:
            return x
        if self.skip_mode == 'add':
            return x + skip
        if self.skip_mode == 'concat':
            return torch.cat([x, skip], dim=1)
        raise ValueError('skip_mode must be one of {none, add, concat}')

    def forward(self, input, frozen_encoder_features=None, frozen_decoder_features=None, return_features: bool = False):
        batch_size = len(input)

        # ================================================================================
        # SECTION 1: SETUP AND VALIDATION
        # ================================================================================
        routed_enc_features, routed_dec_features = self._validate_and_route_frozen_features(
            frozen_encoder_features, frozen_decoder_features
        ) # routed_enc_features: tuple or None; routed_dec_features: tuple or None

        # ================================================================================
        # SECTION 2: ENCODER (Contracting Path: 320 → 160 → 80 → 40 → 20 → 10)
        # ================================================================================
        # Mix frozen features at three spatial scales during encoder:
        # Scale 160 (idx=0), Scale 40 (idx=2), Scale 10 (idx=4)
        # Pattern: Concatenate frozen features → 1x1 conv projects back to base channels
        skips = []
        hidden = input
        
        for idx, block in enumerate(self.contract_blocks):
            hidden = block(hidden)
            
            # Mix frozen encoder features at scales 160, 40, 10
            if self.enable_encoder_mixer and idx in (0, 2, 4):
                inj_idx = {0: 0, 2: 1, 4: 2}[idx]
                key = ('enc_160', 'enc_40', 'enc_10')[inj_idx]
                
                # If frozen channels are configured for this scale, validate and concatenate
                if self.frozen_enc_channels[inj_idx] > 0:
                    frozen_feature = routed_enc_features[inj_idx]
                    frozen_channels_expected = self.frozen_enc_channels[inj_idx]
                    self._validate_feature_shape(frozen_feature, hidden.shape[-2], hidden.shape[-1],
                                                frozen_channels_expected, f'encoder_mix_{inj_idx}')
                    hidden = torch.cat([hidden, frozen_feature], dim=1)  # Concatenate along channel dimension
                
                # Always apply mixer (acts as channel mixer even when no frozen features)
                hidden = self.enc_mixers[key](hidden)
            
            skips.append(hidden) # Skips are appended for every layer because in 'classic' mode we need all of them

        if return_features:
            encoder_feats = [skips[0], skips[2], skips[4]]  # Scales: 160, 40, 10

        # ================================================================================
        # SECTION 3: BOTTLENECK (Processing at smallest spatial scale)
        # ================================================================================
        hidden = self.neck(hidden)

        # ================================================================================
        # SECTION 4: DECODER (Expanding Path: 10 → 20 → 40 → 80 → 160 → 320)
        # ================================================================================
        # Note: In '1x1Conv' mode, frozen features are mixed at scales 10, 40, 160
        #       In 'classic' mode, skip connections merged at scales 10, 20, 40, 80, 160

        # --- Stage 1: Mix/merge at scale 10, then upsample to 20 ---
        routed_dec_feature10 = routed_dec_features[2] if routed_dec_features is not None else None
        hidden, decoder_feat_scale10 = self._inject_and_merge_at_decoder_stage(
            hidden, skips[4], routed_dec_feature10, inject_idx=2, mixer_key='dec_10', 
            return_features=return_features
        )
        hidden = self.expand_blocks[0](hidden)  # 10 → 20

        # --- Stage 2: Merge skip at 20x20 (classic only), then upsample to 40 ---
        if self.skip_handling == 'classic':
            hidden = self._merge_classic(skips[3], hidden)
        hidden = self.expand_blocks[1](hidden)  # 20 → 40

        # --- Stage 3: Mix/merge at scale 40, then upsample to 80 ---
        if self.skip_handling == '1x1Conv':
            routed_dec_feature40 = routed_dec_features[1] if routed_dec_features is not None else None
            hidden, decoder_feat_scale40 = self._inject_and_merge_at_decoder_stage(
                hidden, skips[2], routed_dec_feature40, inject_idx=1, mixer_key='dec_40',
                return_features=return_features
            )
        else:
            # Classic: merge skip before expand
            hidden = self._merge_classic(skips[2], hidden)
            if return_features:
                decoder_feat_scale40 = hidden
        hidden = self.expand_blocks[2](hidden)  # 40 → 80

        # --- Stage 4: Upsample 80 → 160 (skip handled by classic mode only) ---
        if self.skip_handling == 'classic':
            hidden = self._merge_classic( skips[1], hidden)
        hidden = self.expand_blocks[3](hidden)  # 80 → 160

        # --- Stage 5: Mix/merge at scale 160, then upsample to 320 ---

         # Note: a 'classic' mode skip connection is not employed here because you don't want raw skips at final output scale
        routed_dec_feature160 = routed_dec_features[0] if routed_dec_features is not None else None
        hidden, decoder_feat_scale160 = self._inject_and_merge_at_decoder_stage(
            hidden, skips[0], routed_dec_feature160, inject_idx=0, mixer_key='dec_160',
            return_features=return_features
        )
        hidden = self.expand_blocks[4](hidden)  # 160 → 320

        # ================================================================================
        # SECTION 5: POST-PROCESSING (Activation, Normalization, Scaling)
        # ================================================================================
        # Center crop if output exceeds target size
        if hidden.shape[-1] > self.output_size:
            crop_size = self.output_size
            margin = (hidden.shape[-1] - crop_size) // 2
            hidden = hidden[:, :, margin:margin+crop_size, margin:margin+crop_size]

        # Apply final activation (Tanh, Sigmoid, etc.)
        if self.final_activation:
            hidden = self.final_activation(hidden)

        # L1 normalization across spatial dimensions (if enabled)
        if self.normalize:
            hidden = torch.reshape(hidden, (batch_size, self.output_channels, self.output_size**2))
            hidden = nn.functional.normalize(hidden, p=1, dim=2)
            hidden = torch.reshape(hidden, (batch_size, self.output_channels, self.output_size, self.output_size))

        # Apply output scaling (learnable or fixed)
        scale = torch.exp(self.log_output_scale) if self.output_scale_learnable else self.fixed_output_scale
        output = hidden * scale

        # ================================================================================
        # SECTION 6: RETURN OUTPUT (with optional intermediate features)
        # ================================================================================
        if return_features:
            return {
                'output': output,
                'encoder': [encoder_feats[0], encoder_feats[1], encoder_feats[2]],
                'decoder': [decoder_feat_scale160, decoder_feat_scale40, decoder_feat_scale10],
            }
        return output



class Generator_180(nn.Module):
    def __init__(self, config, gen_SI=True, gen_skip_handling: str = '1x1Conv', gen_flow_mode: str = 'coflow', frozen_enc_channels=None, frozen_dec_channels=None, enable_encoder_mixer=True, enable_decoder_mixer=True, scaling_exp=0.7):
        '''
        Encoder-decoder generator with optional skip connections, producing 180x180 output.
        
        Architecture:
            Contracting path: 180→90→45→23→11
            Bottleneck: 1x1, 6x6, or 11x11 spatial size (narrow/medium/wide)
            Expanding path: 11→23→45→90→180
        
        Skip Handling Modes:
            - 'classic': Standard U-Net skip connections (add/concat/none). No feature mixing.
            - '1x1Conv': Advanced mode for frozen flow architectures. Skip connections and frozen
                         features are concatenated, then mixed via 1x1 convolutions at decoder stages.
                         Enables transfer learning from frozen backbone networks.
        
        Flow Modes (1x1Conv only):
            - 'coflow': Frozen encoder features → generator encoder stages,
                        Frozen decoder features → generator decoder stages
            - 'counterflow': Frozen features are swapped (encoder↔decoder)
        
        Frozen Feature Dimensions:
            Format: (channels_at_90, channels_at_23, channels_at_11)
            Specifies expected channel counts from frozen backbone at each scale.
            Set to None or (0,0,0) to disable frozen feature mixing at specific network.
        
        Args:
            config: Dictionary with network hyperparameters and data dimensions:
                - gen_sino_size, gen_sino_channels, gen_image_size, gen_image_channels
                - {SI,IS}_gen_neck: 'narrow'/'medium'/'wide'
                - {SI,IS}_exp_kernel: 3 or 4 (expand kernel size)
                - {SI,IS}_gen_z_dim: Channels in narrowest bottleneck
                - {SI,IS}_gen_hidden_dim: Base channel count
                - {SI,IS}_gen_mult: Channel multiplication factor
                - {SI,IS}_gen_fill: Constant-size conv layers per block (0-3)
                - {SI,IS}_layer_norm: 'batch'/'instance'/'group'/'none'
                - {SI,IS}_pad_mode: 'zeros'/'replicate'
                - {SI,IS}_dropout: True/False
                - {SI,IS}_skip_mode: 'none'/'add'/'concat'
                - {SI,IS}_normalize: Normalize output to L1 norm
                - {SI,IS}_fixedScale or {SI,IS}_learnedScale_init: Output scaling
                - {SI,IS}_gen_final_activ: Final activation (nn.Tanh(), etc.)
            gen_SI: True for sinogram→image, False for image→sinogram
            gen_skip_handling: 'classic' or '1x1Conv'
            gen_flow_mode: 'coflow' or 'counterflow' (only used with 1x1Conv)
            frozen_enc_channels: Tuple (ch_90, ch_23, ch_11) for encoder frozen feature mixing
            frozen_dec_channels: Tuple (ch_90, ch_23, ch_11) for decoder frozen feature mixing
            enable_encoder_mixer: Whether to create encoder mixers (default True in 1x1Conv mode)
            enable_decoder_mixer: Whether to create decoder mixers (default True in 1x1Conv mode)
            scaling_exp: Root exponent used to soften channel growth across stages
                         (e.g., channels scale by mult**(k**scaling_exp))
        
        Example Usage:
            # Classic U-Net:
            gen = Generator_180(config, gen_SI=True, gen_skip_handling='classic')
            output = gen(input)
            
            # Frozen flow (receiving features from frozen backbone):
            gen_trainable = Generator_180(config, gen_SI=True,
                                         gen_skip_handling='1x1Conv',
                                         gen_flow_mode='coflow',
                                         frozen_enc_channels=(64, 128, 256),
                                         frozen_dec_channels=(64, 128, 256))
            output = gen_trainable(input, frozen_encoder_features=enc_feats,
                                  frozen_decoder_features=dec_feats)
        '''
        super(Generator_180, self).__init__()

        # ========================================================================
        # PARSE DIRECTION-SPECIFIC CONFIGURATION (SI vs IS)
        # ========================================================================
        direction_config = self._parse_direction_config(config, gen_SI)

        input_size = direction_config['input_size']
        input_channels = direction_config['input_channels']
        output_size = direction_config['output_size']
        output_channels = direction_config['output_channels']
        neck = direction_config['neck']
        exp_kernel = direction_config['exp_kernel']
        z_dim = direction_config['z_dim']
        hidden_dim = direction_config['hidden_dim']
        fill = direction_config['fill']
        mult = direction_config['mult']
        norm = direction_config['norm']
        pad = direction_config['pad']
        drop = direction_config['drop']
        fixed_scale = direction_config['fixed_scale']
        learned_scale_init = direction_config['learned_scale_init']
        
        self.final_activation = direction_config['final_activation']
        self.normalize = direction_config['normalize']
        self.skip_mode = direction_config['skip_mode']

        # ========================================================================
        # FEATURE MIXING CONFIGURATION (1x1Conv mode only)
        # ========================================================================
        self.skip_handling = gen_skip_handling
        if self.skip_handling not in ('classic', '1x1Conv'):
            raise ValueError('gen_skip_handling must be one of {classic, 1x1Conv}')

        self.flow_mode = gen_flow_mode
        if self.flow_mode not in ('coflow', 'counterflow'):
            raise ValueError('gen_flow_mode must be one of {coflow, counterflow}')

        # Normalize frozen feature channel tuples to (ch_90, ch_23, ch_11) format
        self.frozen_enc_channels = self._normalize_injection_tuple(frozen_enc_channels)
        self.frozen_dec_channels = self._normalize_injection_tuple(frozen_dec_channels)
        
        # Validate: mixing requires 1x1Conv mode
        if self.skip_handling == 'classic' and (any(self.frozen_enc_channels) or any(self.frozen_dec_channels)):
            raise ValueError('Frozen feature mixing requires gen_skip_handling="1x1Conv"')
        
        # Store mixer enablement flags (applies only in 1x1Conv mode)
        self.enable_encoder_mixer = enable_encoder_mixer if self.skip_handling == '1x1Conv' else False
        self.enable_decoder_mixer = enable_decoder_mixer if self.skip_handling == '1x1Conv' else False
        
        # ========================================================================
        # OUTPUT SCALING CONFIGURATION
        # ========================================================================
        self.output_scale_learnable = not bool(self.normalize)
        if self.output_scale_learnable:
            init_scale = float(learned_scale_init if learned_scale_init is not None else fixed_scale)
            self.log_output_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        else:
            init_scale = float(fixed_scale)
            self.register_buffer('fixed_output_scale', torch.tensor(init_scale, dtype=torch.float32))

        self.output_channels = output_channels
        self.output_size = output_size

        in_chan = input_channels
        out_chan = output_channels

        # ========================================================================
        # CHANNEL DIMENSION CALCULATION
        # ========================================================================
        # Root scaling with exponent 0.7 produces gentler channel growth than mult**k
        self.scaling_exp = scaling_exp
        dim_0 = int(hidden_dim * mult ** (0 ** self.scaling_exp))
        dim_1 = int(hidden_dim * mult ** (1 ** self.scaling_exp))
        dim_2 = int(hidden_dim * mult ** (2 ** self.scaling_exp))
        dim_3 = int(hidden_dim * mult ** (3 ** self.scaling_exp))
        dim_4 = int(hidden_dim * mult ** (4 ** self.scaling_exp))
        dim_5 = int(hidden_dim * mult ** (5 ** self.scaling_exp))

        # Channel references for mixing (90, 23, 11 scales)
        self.enc_stage_channels = (dim_0, dim_2, dim_3)
        # Decoder stages at resolutions 11 (pre first upsample), 23 (pre 23->90 upsample), 90 (pre final upsample)
        self.dec_stage_channels = (dim_0, dim_2, dim_3)
        # Skip connection channels at 90, 23, 11 scales
        self.dec_skip_channels = (dim_0, dim_2, dim_3)

        if input_size != 180:
            raise ValueError('This generator is configured for 180x180 inputs and outputs.')

        # ========================================================================
        # BUILD NETWORK ARCHITECTURE
        # ========================================================================

        # Contracting Path: 180 -> 90 -> 45 -> 23 -> 11
        self.contract_blocks = nn.ModuleList([
            contract_block(in_chan, dim_0, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),  # 180->90
            contract_block(dim_0, dim_1, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),    # 90->45
            contract_block(dim_1, dim_2, 3, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),    # 45->23
            contract_block(dim_2, dim_3, 4, stride=2, padding=1, padding_mode=pad, fill=fill, norm=norm, drop=drop),    # 23->11
        ])

        self.neck = self._build_neck(neck, dim_3, dim_4, dim_5, z_dim, pad, fill, norm, drop)
        self.expand_blocks = self._build_expand(exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, pad, fill, norm, drop, self.skip_handling)

        if self.skip_handling == '1x1Conv':
            self._build_mixers()

    # ============================================================================
    # CONFIGURATION HELPERS
    # ============================================================================
    
    def _parse_direction_config(self, config, gen_SI):
        """
        Extract direction-specific configuration (SI vs IS prefixed keys).
        
        Args:
            config: Full configuration dictionary
            gen_SI: True for sinogram→image, False for image→sinogram
        
        Returns:
            Dictionary with normalized keys (without SI_/IS_ prefix)
        """
        if gen_SI:
            return {
                'input_size': config['gen_sino_size'],
                'input_channels': config['gen_sino_channels'],
                'output_size': config['gen_image_size'],
                'output_channels': config['gen_image_channels'],
                'neck': config['SI_gen_neck'],
                'exp_kernel': config['SI_exp_kernel'],
                'z_dim': config['SI_gen_z_dim'],
                'hidden_dim': config['SI_gen_hidden_dim'],
                'fill': config['SI_gen_fill'],
                'mult': config['SI_gen_mult'],
                'norm': config['SI_layer_norm'],
                'pad': config['SI_pad_mode'],
                'drop': config['SI_dropout'],
                'final_activation': config['SI_gen_final_activ'],
                'normalize': config['SI_normalize'],
                'skip_mode': config.get('SI_skip_mode', 'none'),
                'fixed_scale': config.get('SI_fixedScale', 1.0),
                'learned_scale_init': config.get('SI_learnedScale_init'),
            }
        else:
            return {
                'input_size': config['gen_image_size'],
                'input_channels': config['gen_image_channels'],
                'output_size': config['gen_sino_size'],
                'output_channels': config['gen_sino_channels'],
                'neck': config['IS_gen_neck'],
                'exp_kernel': config['IS_exp_kernel'],
                'z_dim': config['IS_gen_z_dim'],
                'hidden_dim': config['IS_gen_hidden_dim'],
                'fill': config['IS_gen_fill'],
                'mult': config['IS_gen_mult'],
                'norm': config['IS_layer_norm'],
                'pad': config['IS_pad_mode'],
                'drop': config['IS_dropout'],
                'final_activation': config['IS_gen_final_activ'],
                'normalize': config['IS_normalize'],
                'skip_mode': config.get('IS_skip_mode', 'none'),
                'fixed_scale': config.get('IS_fixedScale', 1.0),
                'learned_scale_init': config.get('IS_learnedScale_init'),
            }
    
    def _normalize_injection_tuple(self, cfg):
        """
        Normalize injection channel tuple to (ch_90, ch_23, ch_11) format.
        Ensures that if None is provided, it defaults to (0,0,0).
        
        Args:
            cfg: None, or tuple/list of 3 integers
        
        Returns:
            Tuple of 3 integers (0s if None provided)
        """
        if cfg is None:
            return (0, 0, 0)
        if len(cfg) != 3:
            raise ValueError('Injection tuples must have three entries (90, 23, 11 scales).')
        return tuple(int(x) for x in cfg)

    # ============================================================================
    # ARCHITECTURE BUILDERS
    # ============================================================================

    def _build_neck(self, neck, dim_3, dim_4, dim_5, z_dim, pad, fill, norm, drop):
        """
        Build the bottleneck network architecture between encoder and decoder.
        
        Controls information flow at the network's narrowest point. Three modes control
        spatial compression and capacity:
        - 'narrow': 11→6→3→1→3→6→11 (aggressive compression, dense bottleneck)
        - 'medium': 11→6→6→6→6→6→11 (moderate compression, constant 5x5 processing)
        - 'wide': 11→11→11→11→11→11 (no compression, spatial information preserved)
        
        Args:
            neck: Bottleneck mode {'narrow', 'medium', 'wide'}
            dim_3: Channels at encoder output (scale 11)
            dim_4: Channels at intermediate bottleneck stage
            dim_5: Channels at deepest bottleneck stage
            z_dim: Channels at 1x1 spatial bottleneck (narrow mode only)
            pad: Padding mode for convolutions
            fill: Number of constant-size conv layers per block
            norm: Normalization type
            drop: Whether to use dropout
        
        Returns:
            nn.Sequential: Bottleneck network module
        """
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

    def _build_expand(self, exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, pad, fill, norm, drop, skip_handling):
        """
        Build the decoder (expanding path) from bottleneck to output resolution.
        
        Creates 4 upsampling stages: 11→23→45→90→180. Each stage uses transposed
        convolutions with optional fill layers. Input channels are adjusted based on
        skip connection mode (doubled for concat mode in classic skip handling).
        
        Args:
            exp_kernel: Kernel size for transposed convolutions {3, 4}
            out_chan: Output channels (final image/sinogram channels)
            dim_0: Channels at scale 90
            dim_1: Channels at scale 45
            dim_2: Channels at scale 23
            dim_3: Channels at scale 11
                        skip_handling: Skip connection mode {'classic', '1x1Conv'}
            pad: Padding mode (decoder uses 'replicate')
            fill: Number of constant-size conv layers per block
            norm: Normalization type
            drop: Whether to use dropout
        
        Returns:
            nn.ModuleList: List of 4 expand blocks for upsampling stages
        """
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
            raise ValueError('exp_kernel must be 3 or 4 for Generator_180')

        def in_ch(base):
            """
            Calculate input channels for expand block.
            In classic mode with concat: doubles channels to accommodate skip connection
            In 1x1Conv mode: base channels (merging handled by mixers)
            """
            if skip_handling == '1x1Conv':
                return base
            return base * 2 if self.skip_mode == 'concat' else base

        blocks = nn.ModuleList()
        blocks.append(expand_block(in_ch(dim_3), dim_2, stage_params[0][0], stage_params[0][1], stage_params[0][2], stage_params[0][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_2), dim_1, stage_params[1][0], stage_params[1][1], stage_params[1][2], stage_params[1][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_1), dim_0, stage_params[2][0], stage_params[2][1], stage_params[2][2], stage_params[2][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_0), out_chan, stage_params[3][0], stage_params[3][1], stage_params[3][2], stage_params[3][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop, final_layer=True))
        return blocks

    def _build_mixers(self):
        """
        Build 1x1 convolutional projection layers for feature mixing in 1x1Conv mode.
        
        Creates two sets of mixers:
        - Encoder mixers: Mix frozen encoder features at scales 90, 23, 11 (if enabled)
        - Decoder mixers: Mix skip connections + frozen decoder features (always enabled in 1x1Conv)
        
        Each mixer concatenates multiple feature sources (base features + optional skip + optional frozen),
        then projects back to base channel count via 1x1 conv. Always created when enabled.
                
        Returns:
            None (creates self.enc_mixers and self.dec_mixers ModuleDicts)
        """
        def _make_proj(in_ch, out_ch):
            return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        # Encoder mixers (created if enabled)
        enc_keys = ['enc_90', 'enc_23', 'enc_11']
        enc_chs = self.enc_stage_channels
        self.enc_mixers = nn.ModuleDict()
        if self.enable_encoder_mixer:
            for key, base_ch, frozen_ch in zip(enc_keys, enc_chs, self.frozen_enc_channels):
                # Mixer handles base channels + any frozen channels at this scale
                self.enc_mixers[key] = _make_proj(base_ch + frozen_ch, base_ch)

        # Decoder mixers (always created in 1x1Conv mode)
        dec_keys = ['dec_90', 'dec_23', 'dec_11']
        dec_chs = self.dec_stage_channels
        skip_chs = self.dec_skip_channels
        self.dec_mixers = nn.ModuleDict()
        for key, base_ch, skip_ch, frozen_ch in zip(dec_keys, dec_chs, skip_chs, self.frozen_dec_channels):
            total_in = base_ch + frozen_ch
            if self.skip_mode != 'none':
                total_in += skip_ch
            self.dec_mixers[key] = _make_proj(total_in, base_ch)

    # ============================================================================
    # FORWARD PASS HELPERS
    # ============================================================================
    
    def _validate_and_route_frozen_features(self, frozen_encoder_features, frozen_decoder_features):
        """
        Validate and route frozen features based on skip_handling and flow_mode.
        
        Args:
            frozen_encoder_features: Encoder features from frozen network (or None)
            frozen_decoder_features: Decoder features from frozen network (or None)
        
        Returns:
            (routed_encoder_features, routed_decoder_features) as tuples or None
        """

        # Route features based on flow mode
        routed_enc_features = frozen_encoder_features
        routed_dec_features = frozen_decoder_features
        if self.flow_mode == 'counterflow':
            # Counterflow: swap encoder and decoder features
            routed_enc_features, routed_dec_features = routed_dec_features, routed_enc_features

        # Convert to tuples for indexing
        routed_enc_features = tuple(routed_enc_features) if routed_enc_features is not None else None
        routed_dec_features = tuple(routed_dec_features) if routed_dec_features is not None else None

        # Validate: frozen encoder features require encoder mixer to be enabled
        if routed_enc_features is not None and not self.enable_encoder_mixer:
            raise ValueError('Frozen encoder features provided but enable_encoder_mixer=False')
        
        # Validate: frozen decoder features require decoder mixer to be enabled
        if routed_dec_features is not None and not self.enable_decoder_mixer:
            raise ValueError('Frozen decoder features provided but enable_decoder_mixer=False')

        # Validate: classic mode cannot accept frozen features
        if self.skip_handling == 'classic' and (frozen_encoder_features is not None or frozen_decoder_features is not None):
            raise ValueError('Frozen features provided but gen_skip_handling is classic.')

        return routed_enc_features, routed_dec_features
    
    def _validate_feature_shape(self, tensor, target_h, target_w, expected_c, label):
        """
        Validate that a feature tensor has expected spatial size and channel count.
        
        Args:
            tensor: Feature tensor to validate
            target_h: Expected height
            target_w: Expected width
            expected_c: Expected channel count
            label: Descriptive label for error messages
        
        Raises:
            ValueError: If shape doesn't match expectations
        """
        if tensor is None:
            raise ValueError(f'Missing tensor for {label}')
        h, w = tensor.shape[-2:]
        if h != target_h or w != target_w:
            raise ValueError(f'Shape mismatch for {label}: expected {target_h}x{target_w}, got {h}x{w}')
        if tensor.shape[1] != expected_c:
            raise ValueError(f'Channel mismatch for {label}: expected {expected_c}, got {tensor.shape[1]}')
    
    def _inject_and_merge_at_decoder_stage(self, hidden, skip_tensor, frozen_feature,
                                           inject_idx, mixer_key, return_features):
        """
        Mix skip connection and frozen decoder features at a decoder stage.
        
        Handles both classic mode (simple merge) and 1x1Conv mode (concat + projection).
        
        Args:
            hidden: Current decoder output tensor
            skip_tensor: Skip connection from encoder (always passed, even if not used)
            frozen_feature: Frozen feature to mix (or None)
            inject_idx: Index into frozen_dec_channels (0, 1, or 2)
            mixer_key: Key for dec_mixers ModuleDict
            return_features: Whether to capture features for return
        
        Returns:
            (updated_hidden, optional_captured_feature)
        """
        captured_feature = None
        
        if self.skip_handling == '1x1Conv':
            # 1x1Conv mode: concatenate decoder output + optional skip + optional frozen feature
            parts = [hidden]
            if self.skip_mode != 'none':
                parts.append(skip_tensor)

            if self.frozen_dec_channels[inject_idx] > 0:
                frozen_channels_expected = self.frozen_dec_channels[inject_idx]
                self._validate_feature_shape(frozen_feature, skip_tensor.shape[-2],
                                            skip_tensor.shape[-1], frozen_channels_expected,
                                            f'decoder_mix_{mixer_key}')
                parts.append(frozen_feature)

            # Concatenate and project back to base channels
            hidden = torch.cat(parts, dim=1)
            hidden = self.dec_mixers[mixer_key](hidden)

            # Capture features AFTER mixing
            if return_features:
                captured_feature = hidden
        else:
            # Classic mode: simple skip connection merge
            if return_features:
                captured_feature = hidden # Capture BEFORE merging because otherwise the channel number would be 2X larger
            hidden = self._merge_classic(skip_tensor, hidden)
        
        return hidden, captured_feature

    def _merge_classic(self, skip, x):
        """
        Merge skip connection with decoder output using configured skip_mode.
        Used in classic mode only; 1x1Conv mode uses _inject_and_merge_at_decoder_stage.
        
        Args:
            skip: Skip connection tensor from encoder
            x: Current decoder output tensor
        
        Returns:
            Merged tensor (add, concat, or passthrough if skip_mode='none')
        """
        if self.skip_mode == 'none' or skip is None:
            return x
        if self.skip_mode == 'add':
            return x + skip
        if self.skip_mode == 'concat':
            return torch.cat([x, skip], dim=1)
        raise ValueError('skip_mode must be one of {none, add, concat}')

    def forward(self, input, frozen_encoder_features=None, frozen_decoder_features=None, return_features: bool = False):
        batch_size = len(input)

        # ================================================================================
        # SECTION 1: SETUP AND VALIDATION
        # ================================================================================
        routed_enc_features, routed_dec_features = self._validate_and_route_frozen_features(
            frozen_encoder_features, frozen_decoder_features
        ) # routed_enc_features: tuple or None; routed_dec_features: tuple or None

        # ================================================================================
        # SECTION 2: ENCODER (Contracting Path: 180 → 90 → 45 → 23 → 11)
        # ================================================================================
        # Mix frozen features at three spatial scales during encoder:
        # Scale 90 (idx=0), Scale 23 (idx=2), Scale 11 (idx=3)
        # Pattern: Concatenate frozen features → 1x1 conv projects back to base channels
        skips = []
        hidden = input
        
        for idx, block in enumerate(self.contract_blocks):
            hidden = block(hidden)
            
            # Mix frozen encoder features at scales 90, 23, 11
            if self.enable_encoder_mixer and idx in (0, 2, 3):
                inj_idx = {0: 0, 2: 1, 3: 2}[idx]
                key = ('enc_90', 'enc_23', 'enc_11')[inj_idx]
                
                # If frozen channels are configured for this scale, validate and concatenate
                if self.frozen_enc_channels[inj_idx] > 0:
                    frozen_feature = routed_enc_features[inj_idx]
                    frozen_channels_expected = self.frozen_enc_channels[inj_idx]
                    self._validate_feature_shape(frozen_feature, hidden.shape[-2], hidden.shape[-1],
                                                frozen_channels_expected, f'encoder_mix_{inj_idx}')
                    hidden = torch.cat([hidden, frozen_feature], dim=1)  # Concatenate along channel dimension
                
                # Always apply mixer (acts as channel mixer even when no frozen features)
                hidden = self.enc_mixers[key](hidden)
            
            skips.append(hidden) # Skips are appended for every layer because in 'classic' mode we need all of them

        if return_features:
            encoder_feats = [skips[0], skips[2], skips[3]]  # Scales: 90, 23, 11

        # ================================================================================
        # SECTION 3: BOTTLENECK (Processing at smallest spatial scale)
        # ================================================================================
        hidden = self.neck(hidden)

        # ================================================================================
        # SECTION 4: DECODER (Expanding Path: 11 → 23 → 45 → 90 → 180)
        # ================================================================================
        # Note: In '1x1Conv' mode, frozen features are mixed at scales 11, 23, 90
        #       In 'classic' mode, skip connections merged at scales 11, 23, 45, 90

        # --- Stage 1: Mix/merge at scale 11, then upsample to 23 ---
        routed_dec_feature11 = routed_dec_features[2] if routed_dec_features is not None else None
        hidden, decoder_feat_scale11 = self._inject_and_merge_at_decoder_stage(
            hidden, skips[3], routed_dec_feature11, inject_idx=2, mixer_key='dec_11', 
            return_features=return_features
        )
        hidden = self.expand_blocks[0](hidden)  # 11 → 23

        # --- Stage 2: Mix/merge at scale 23, then upsample to 45 ---
        if self.skip_handling == '1x1Conv':
            routed_dec_feature23 = routed_dec_features[1] if routed_dec_features is not None else None
            hidden, decoder_feat_scale23 = self._inject_and_merge_at_decoder_stage(
                hidden, skips[2], routed_dec_feature23, inject_idx=1, mixer_key='dec_23',
                return_features=return_features
            )
        else:
            # Classic: merge skip before expand
            hidden = self._merge_classic(skips[2], hidden)
            if return_features:
                decoder_feat_scale23 = hidden
        hidden = self.expand_blocks[1](hidden)  # 23 → 45

        # --- Stage 3: Upsample 45 → 90 (skip handled by classic mode only) ---
        if self.skip_handling == 'classic':
            hidden = self._merge_classic(skips[1], hidden)
        hidden = self.expand_blocks[2](hidden)  # 45 → 90

        # --- Stage 4: Mix/merge at scale 90, then upsample to 180 ---
        routed_dec_feature90 = routed_dec_features[0] if routed_dec_features is not None else None
        hidden, decoder_feat_scale90 = self._inject_and_merge_at_decoder_stage(
            hidden, skips[0], routed_dec_feature90, inject_idx=0, mixer_key='dec_90',
            return_features=return_features
        )
        hidden = self.expand_blocks[3](hidden)  # 90 → 180

        # ================================================================================
        # SECTION 5: POST-PROCESSING (Activation, Normalization, Scaling)
        # ================================================================================
        # Center crop if output exceeds target size
        if hidden.shape[-1] > self.output_size:
            crop_size = self.output_size
            margin = (hidden.shape[-1] - crop_size) // 2
            hidden = hidden[:, :, margin:margin+crop_size, margin:margin+crop_size]

        # Apply final activation (Tanh, Sigmoid, etc.)
        if self.final_activation:
            hidden = self.final_activation(hidden)

        # L1 normalization across spatial dimensions (if enabled)
        if self.normalize:
            hidden = torch.reshape(hidden, (batch_size, self.output_channels, self.output_size**2))
            hidden = nn.functional.normalize(hidden, p=1, dim=2)
            hidden = torch.reshape(hidden, (batch_size, self.output_channels, self.output_size, self.output_size))

        # Apply output scaling (learnable or fixed)
        scale = torch.exp(self.log_output_scale) if self.output_scale_learnable else self.fixed_output_scale
        output = hidden * scale

        # ================================================================================
        # SECTION 6: RETURN OUTPUT (with optional intermediate features)
        # ================================================================================
        if return_features:
            return {
                'output': output,
                'encoder': [encoder_feats[0], encoder_feats[1], encoder_feats[2]],
                'decoder': [decoder_feat_scale90, decoder_feat_scale23, decoder_feat_scale11],
            }
        return output





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