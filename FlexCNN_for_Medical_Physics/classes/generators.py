import torch
from torch import nn


#############################
##### Generator Classes #####
#############################

class Generator_288(nn.Module):
    def __init__(self, config, gen_SI=True, gen_skip_handling: str = 'classic', gen_flow_mode: str = 'coflow', enc_inject_channels=None, dec_inject_channels=None, scaling_exp=0.7):
        '''
        Encoder-decoder generator with optional skip connections, producing 288x288 output.
        
        Architecture:
            Contracting path: 288→144→72→36→18→9
            Bottleneck: 1x1, 5x5, or 9x9 spatial size (narrow/medium/wide)
            Expanding path: 9→18→36→72→144→288
        
        Skip Handling Modes:
            - 'classic': Standard U-Net skip connections (add/concat/none). No feature injection.
                         ~95% of use cases. Default mode for standard training.
            - '1x1Conv': Advanced mode for frozen flow architectures. Skip connections and frozen
                         features are concatenated, then projected via 1x1 convolutions at decoder stages.
                         Enables transfer learning from frozen backbone networks.
        
        Flow Modes (1x1Conv only):
            - 'coflow': Frozen encoder features → decoder encoder stages,
                        Frozen decoder features → decoder decoder stages
            - 'counterflow': Frozen features are swapped (encoder↔decoder)
        
        Injection Tuples:
            Format: (channels_at_144, channels_at_36, channels_at_9)
            Set to None or (0,0,0) to disable injection at specific network.
        
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
            enc_inject_channels: Tuple (ch_144, ch_36, ch_9) for encoder injection
            dec_inject_channels: Tuple (ch_144, ch_36, ch_9) for decoder injection
        
        Example Usage:
            # Classic U-Net:
            gen = Generator_288(config, gen_SI=True)
            output = gen(input)
            
            # Frozen flow (receiving features from frozen backbone):
            gen_trainable = Generator_288(config, gen_SI=True,
                                         gen_skip_handling='1x1Conv',
                                         gen_flow_mode='coflow',
                                         enc_inject_channels=(64, 128, 256),
                                         dec_inject_channels=(64, 128, 256))
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
        # FEATURE INJECTION CONFIGURATION (1x1Conv mode only)
        # ========================================================================
        self.skip_handling = gen_skip_handling
        if self.skip_handling not in ('classic', '1x1Conv'):
            raise ValueError('gen_skip_handling must be one of {classic, 1x1Conv}')

        self.flow_mode = gen_flow_mode
        if self.flow_mode not in ('coflow', 'counterflow'):
            raise ValueError('gen_flow_mode must be one of {coflow, counterflow}')

        # Normalize injection tuples to (ch_144, ch_36, ch_9) format
        self.enc_inject_channels = self._normalize_injection_tuple(enc_inject_channels)
        self.dec_inject_channels = self._normalize_injection_tuple(dec_inject_channels)
        
        # Validate: injection requires 1x1Conv mode
        if self.skip_handling == 'classic' and (any(self.enc_inject_channels) or any(self.dec_inject_channels)):
            raise ValueError('Injection requires gen_skip_handling="1x1Conv"')
        
        # Set flags for conditional injection logic
        self.enable_encoder_inject = self.skip_handling == '1x1Conv' and any(self.enc_inject_channels)
        self.enable_decoder_inject = self.skip_handling == '1x1Conv' and any(self.dec_inject_channels)
        
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

        # Channel references for injection (144,36,9 scales)
        self.enc_stage_channels = (dim_0, dim_2, dim_4)
        # Decoder stages at resolutions 9 (pre first upsample), 36 (pre 36->144 upsample), 144 (pre final upsample)
        self.dec_stage_channels = (dim_0, dim_2, dim_4)
        self.dec_skip_channels = (dim_0, dim_2, dim_4)

        if self.skip_handling == '1x1Conv':
            self._build_injectors(pad, norm, drop)

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
            dim_0: Channels at scale 144/288
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

    def _build_injectors(self, pad, norm, drop):
        """
        Build 1x1 convolutional projection layers for feature injection in 1x1Conv mode.
        
        Creates two sets of injectors:
        - Encoder injectors: Merge frozen encoder features at scales 144, 36, 9
        - Decoder injectors: Merge skip connections + frozen decoder features (always created)
        
        Each injector concatenates multiple feature sources (base features + skip + frozen),
        then projects back to base channel count via 1x1 conv. Only built when skip_handling='1x1Conv'.
        
        Args:
            pad: Padding mode (unused, kept for consistency)
            norm: Normalization type (unused, kept for consistency)
            drop: Dropout flag (unused, kept for consistency)
        
        Returns:
            None (creates self.enc_injectors and self.dec_injectors ModuleDicts)
        """
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

    # ============================================================================
    # FORWARD PASS HELPERS
    # ============================================================================
    
    def _setup_and_validate_frozen_features(self, frozen_encoder_features, frozen_decoder_features):
        """
        Validate and route frozen features based on skip_handling and flow_mode.
        
        Args:
            frozen_encoder_features: Encoder features from frozen network (or None)
            frozen_decoder_features: Decoder features from frozen network (or None)
        
        Returns:
            (routed_encoder_features, routed_decoder_features) as tuples or None
        """
        # Validate: classic mode cannot accept frozen features
        if self.skip_handling == 'classic' and (frozen_encoder_features is not None or frozen_decoder_features is not None):
            raise ValueError('Frozen features provided but gen_skip_handling is classic.')

        # Route features based on flow mode
        routed_enc_features = frozen_encoder_features
        routed_dec_features = frozen_decoder_features
        if self.flow_mode == 'counterflow':
            # Counterflow: swap encoder and decoder features
            routed_enc_features, routed_dec_features = routed_dec_features, routed_enc_features

        # Convert to tuples for indexing
        routed_enc_features = tuple(routed_enc_features) if routed_enc_features is not None else None
        routed_dec_features = tuple(routed_dec_features) if routed_dec_features is not None else None

        # Validate: if injection is enabled, features must be provided
        if self.skip_handling == '1x1Conv':
            if any(self.enc_inject_channels) and routed_enc_features is None:
                raise ValueError('Encoder injection requested but frozen_encoder_features not provided.')
            if self.enable_decoder_inject and routed_dec_features is None and any(self.dec_inject_channels):
                raise ValueError('Decoder injection requested but frozen_decoder_features not provided.')

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
                                           inject_idx, injector_key, return_features):
        """
        Merge skip connection and inject frozen features at a decoder stage.
        
        Handles both classic mode (simple merge) and 1x1Conv mode (concat + projection).
        
        Args:
            hidden: Current decoder output tensor
            skip_tensor: Skip connection from encoder
            frozen_feature: Frozen feature to inject (or None for classic mode)
            inject_idx: Index into dec_inject_channels (0, 1, or 2)
            injector_key: Key for dec_injectors ModuleDict
            return_features: Whether to capture features for return
        
        Returns:
            (updated_hidden, optional_captured_feature)
        """
        captured_feature = None
        
        if self.skip_handling == '1x1Conv':
            # 1x1Conv mode: concatenate skip + frozen feature, then project
            inject_channels_expected = self.dec_inject_channels[inject_idx]
            self._validate_feature_shape(frozen_feature, skip_tensor.shape[-2], 
                                        skip_tensor.shape[-1], inject_channels_expected,
                                        f'decoder_inject_{injector_key}')
            
            # Build concatenation list: decoder output + optional skip + frozen feature
            parts = [hidden]
            if self.skip_mode != 'none':
                parts.append(skip_tensor)
            parts.append(frozen_feature)
            
            # Concatenate and project back to base channels
            hidden = torch.cat(parts, dim=1)
            hidden = self.dec_injectors[injector_key](hidden)
            
            # Capture features AFTER injection
            if return_features:
                captured_feature = hidden
        else:
            # Classic mode: simple skip connection merge
            if return_features:
                captured_feature = hidden
            hidden = self._merge(skip_tensor, hidden)
        
        return hidden, captured_feature
    
    def _merge(self, skip, x):
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
        routed_enc_features, routed_dec_features = self._setup_and_validate_frozen_features(
            frozen_encoder_features, frozen_decoder_features
        )

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
            
            # Inject frozen encoder features at scales 144, 36, 9
            if self.skip_handling == '1x1Conv' and idx in (0, 2, 4):
                inj_idx = {0: 0, 2: 1, 4: 2}[idx]
                inject_channels_expected = self.enc_inject_channels[inj_idx]
                frozen_feature = routed_enc_features[inj_idx]
                
                self._validate_feature_shape(frozen_feature, hidden.shape[-2], hidden.shape[-1],
                                            inject_channels_expected, f'encoder_inject_{inj_idx}')
                
                key = ('enc_144', 'enc_36', 'enc_9')[inj_idx]
                hidden = torch.cat([hidden, frozen_feature], dim=1)  # Concatenate along channel dimension
                hidden = self.enc_injectors[key](hidden)              # Project back to base channels
            
            skips.append(hidden)

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

        # --- Stage 1: Injection/merge at scale 9, then upsample to 18 ---
        frozen_dec_feat = routed_dec_features[2] if routed_dec_features is not None else None
        hidden, decoder_feat_scale9 = self._inject_and_merge_at_decoder_stage(
            hidden, skips[4], frozen_dec_feat, inject_idx=2, injector_key='dec_9', 
            return_features=return_features
        )
        hidden = self.expand_blocks[0](hidden)  # 9 → 18

        # --- Stage 2: Upsample 18 → 36 (skip handled by classic mode only) ---
        if self.skip_handling == 'classic':
            hidden = self._merge(skips[3], hidden)
        hidden = self.expand_blocks[1](hidden)  # 18 → 36

        # --- Stage 3: Upsample 36 → 72 (skip handled by classic mode only) ---
        if self.skip_handling == 'classic':
            hidden = self._merge(skips[2], hidden)
        hidden = self.expand_blocks[2](hidden)  # 36 → 72

        # --- Stage 4: Injection/merge at scale 72, then upsample to 144 ---
        frozen_dec_feat = routed_dec_features[1] if routed_dec_features is not None else None
        hidden, decoder_feat_scale36 = self._inject_and_merge_at_decoder_stage(
            hidden, skips[1], frozen_dec_feat, inject_idx=1, injector_key='dec_36',
            return_features=return_features
        )
        hidden = self.expand_blocks[3](hidden)  # 72 → 144

        # --- Stage 5: Injection/merge at scale 144, then upsample to 288 ---
        frozen_dec_feat = routed_dec_features[0] if routed_dec_features is not None else None
        hidden, decoder_feat_scale144 = self._inject_and_merge_at_decoder_stage(
            hidden, skips[0], frozen_dec_feat, inject_idx=0, injector_key='dec_144',
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
    def __init__(self, config, gen_SI=True, scaling_exp=0.7):
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

            neck = config['SI_gen_neck']
            exp_kernel = config['SI_exp_kernel']
            z_dim = config['SI_gen_z_dim']
            hidden_dim = config['SI_gen_hidden_dim']
            fill = config['SI_gen_fill']
            mult = config['SI_gen_mult']
            norm = config['SI_layer_norm']
            pad = config['SI_pad_mode']
            drop = config['SI_dropout']
            skip_mode = config.get('SI_skip_mode', 'none')
            fixed_scale = config.get('SI_fixedScale', 1.0)
            learned_scale_init = config.get('SI_learnedScale_init')

            self.final_activation = config['SI_gen_final_activ']
            self.normalize = config['SI_normalize']
        else:
            input_size = config['gen_image_size']
            input_channels = config['gen_image_channels']
            output_size = config['gen_sino_size']
            output_channels = config['gen_sino_channels']

            neck = config['IS_gen_neck']
            exp_kernel = config['IS_exp_kernel']
            z_dim = config['IS_gen_z_dim']
            hidden_dim = config['IS_gen_hidden_dim']
            fill = config['IS_gen_fill']
            mult = config['IS_gen_mult']
            norm = config['IS_layer_norm']
            pad = config['IS_pad_mode']
            drop = config['IS_dropout']
            skip_mode = config.get('IS_skip_mode', 'none')
            fixed_scale = config.get('IS_fixedScale', 1.0)
            learned_scale_init = config.get('IS_learnedScale_init')

            self.final_activation = config['IS_gen_final_activ']
            self.normalize = config['IS_normalize']

        self.skip_mode = skip_mode

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

        # Root scaling exponent for channel growth
        self.scaling_exp = scaling_exp
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

    def _build_expand(self, exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, dim_4, pad, fill, norm, drop):
        """
        Build the decoder (expanding path) from bottleneck to output resolution.
        
        Creates 5 upsampling stages: 10→20→40→80→160→320. Each stage uses transposed
        convolutions with optional fill layers. Input channels are adjusted based on
        skip connection mode (doubled for concat mode).
        
        Args:
            exp_kernel: Kernel size for transposed convolutions {3, 4}
            out_chan: Output channels (final image/sinogram channels)
            dim_0: Channels at scale 160/320
            dim_1: Channels at scale 80
            dim_2: Channels at scale 40
            dim_3: Channels at scale 20
            dim_4: Channels at scale 10
            pad: Padding mode (decoder uses 'replicate')
            fill: Number of constant-size conv layers per block
            norm: Normalization type
            drop: Whether to use dropout
        
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
            return base * 2 if self.skip_mode == 'concat' else base

        blocks = nn.ModuleList()
        blocks.append(expand_block(in_ch(dim_4), dim_3, stage_params[0][0], stage_params[0][1], stage_params[0][2], stage_params[0][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_3), dim_2, stage_params[1][0], stage_params[1][1], stage_params[1][2], stage_params[1][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_2), dim_1, stage_params[2][0], stage_params[2][1], stage_params[2][2], stage_params[2][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_1), dim_0, stage_params[3][0], stage_params[3][1], stage_params[3][2], stage_params[3][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop))
        blocks.append(expand_block(in_ch(dim_0), out_chan, stage_params[4][0], stage_params[4][1], stage_params[4][2], stage_params[4][3], padding_mode='replicate', fill=fill, norm=norm, drop=drop, final_layer=True))
        return blocks

    def _merge(self, skip, x):
        """
        Merge skip connection with decoder output based on configured skip_mode.
        
        Supports three merging strategies:
        - 'none': No skip connection, return decoder output unchanged
        - 'add': Element-wise addition (residual connection)
        - 'concat': Channel-wise concatenation (U-Net style)
        
        Args:
            skip: Skip connection tensor from encoder (or None)
            x: Current decoder output tensor
        
        Returns:
            torch.Tensor: Merged tensor
        """
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
    def __init__(self, config, gen_SI=True, scaling_exp=0.7):
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

            neck = config['SI_gen_neck']
            exp_kernel = config['SI_exp_kernel']
            z_dim = config['SI_gen_z_dim']
            hidden_dim = config['SI_gen_hidden_dim']
            fill = config['SI_gen_fill']
            mult = config['SI_gen_mult']
            norm = config['SI_layer_norm']
            pad = config['SI_pad_mode']
            drop = config['SI_dropout']
            skip_mode = config.get('SI_skip_mode', 'none')
            fixed_scale = config.get('SI_fixedScale', 1.0)
            learned_scale_init = config.get('SI_learnedScale_init')

            self.final_activation = config['SI_gen_final_activ']
            self.normalize = config['SI_normalize']
        else:
            input_size = config['gen_image_size']
            input_channels = config['gen_image_channels']
            output_size = config['gen_sino_size']
            output_channels = config['gen_sino_channels']

            neck = config['IS_gen_neck']
            exp_kernel = config['IS_exp_kernel']
            z_dim = config['IS_gen_z_dim']
            hidden_dim = config['IS_gen_hidden_dim']
            fill = config['IS_gen_fill']
            mult = config['IS_gen_mult']
            norm = config['IS_layer_norm']
            pad = config['IS_pad_mode']
            drop = config['IS_dropout']
            skip_mode = config.get('IS_skip_mode', 'none')
            fixed_scale = config.get('IS_fixedScale', 1.0)
            learned_scale_init = config.get('IS_learnedScale_init')

            self.final_activation = config['IS_gen_final_activ']
            self.normalize = config['IS_normalize']

        self.skip_mode = skip_mode

        self.output_scale_learnable = not bool(self.normalize)  # If learning scale only when not normalizing
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

        # Root scaling exponent for channel growth
        self.scaling_exp = scaling_exp
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

    def _build_expand(self, exp_kernel, out_chan, dim_0, dim_1, dim_2, dim_3, pad, fill, norm, drop):
        """
        Build the decoder (expanding path) from bottleneck to output resolution.
        
        Creates 4 upsampling stages: 11→23→45→90→180. Each stage uses transposed
        convolutions with optional fill layers. Input channels are adjusted based on
        skip connection mode (doubled for concat mode).
        
        Args:
            exp_kernel: Kernel size for transposed convolutions {3, 4}
            out_chan: Output channels (final image/sinogram channels)
            dim_0: Channels at scale 90/180
            dim_1: Channels at scale 45
            dim_2: Channels at scale 23
            dim_3: Channels at scale 11
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
        """
        Merge skip connection with decoder output based on configured skip_mode.
        
        Supports three merging strategies:
        - 'none': No skip connection, return decoder output unchanged
        - 'add': Element-wise addition (residual connection)
        - 'concat': Channel-wise concatenation (U-Net style)
        
        Args:
            skip: Skip connection tensor from encoder (or None)
            x: Current decoder output tensor
        
        Returns:
            torch.Tensor: Merged tensor
        """
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