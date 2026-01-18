import os
import time
import torch
import pandas as pd
import logging
from torch import nn
from torch.utils.data import DataLoader

from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataSet
from FlexCNN_for_Medical_Physics.classes.losses import HybridLoss
from FlexCNN_for_Medical_Physics.functions.helper.timing import display_times
from FlexCNN_for_Medical_Physics.functions.helper.metrics_wrappers import calculate_metric
from FlexCNN_for_Medical_Physics.functions.helper.metrics import SSIM, MSE
from FlexCNN_for_Medical_Physics.functions.helper.weights_init import weights_init_he
from FlexCNN_for_Medical_Physics.functions.helper.displays_and_reports import (
    compute_display_step,
    get_tune_session,
)
from FlexCNN_for_Medical_Physics.functions.main_run_functions.train_utils import (
    collate_nested,
    create_generator,
    create_optimizer,
    build_checkpoint_dict,
    save_checkpoint,
    log_tune_debug,
    report_tune_metrics,
    visualize_train,
    visualize_test,
    visualize_mode,
    route_batch_inputs,
    generate_reconstructions_for_visualization,
    compute_test_metrics,
    init_checkpoint_state,
    apply_feature_injection,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_frozen_state_dict(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint.get('gen_state_dict', checkpoint)


def _init_feat_scales(injection_feats, init_value: float = 0.1):
    scale_params = nn.ParameterList()
    for feat in injection_feats:
        c = feat.shape[1]
        param = nn.Parameter(torch.full((1, c, 1, 1), init_value, device=feat.device))
        scale_params.append(param)
    return scale_params


def _build_feat_scale_dict(frozen_encoder_feats, frozen_decoder_feats, flow_mode: str, init_value: float = 0.1):
    if flow_mode == 'coflow':
        enc_sources = frozen_encoder_feats
        dec_sources = frozen_decoder_feats
    else:
        enc_sources = frozen_decoder_feats
        dec_sources = frozen_encoder_feats
    scales = {
        'encoder': _init_feat_scales(enc_sources, init_value),
        'decoder': _init_feat_scales(dec_sources, init_value),
    }
    return scales


def run_trainable_frozen_flow(config, paths, settings):
    """
    Train/test/visualize with frozen attenuation network guiding a trainable activity network.
    """
    run_mode = settings['run_mode']
    device = settings['device']
    tune_debug = settings.get('tune_debug', False)
    if tune_debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"[TUNE_DEBUG] Entering run_trainable_frozen_flow; device: {device}")

    flow_mode = 'coflow' if config['network_type'] == 'FROZEN_COFLOW' else 'counterflow'
    train_SI_act = True  # activity network is always sino->image

    augment = settings['augment']
    shuffle = settings['shuffle']
    offset = settings['offset']
    num_examples = settings['num_examples']
    sample_division = settings['sample_division']

    num_epochs = settings['num_epochs']
    load_state = settings['load_state']
    save_state = settings['save_state']
    show_times = settings['show_times']

    if run_mode == 'tune':
        tune_dataframe_fraction = settings['tune_dataframe_fraction']
        tune_max_t = settings['tune_max_t']
        tune_restore = settings['tune_restore']
        tune_dataframe_path = paths['tune_dataframe_path']
    checkpoint_path = paths['checkpoint_path']

    if 'batch_base2_exponent' in config and run_mode in ('tune', 'train'):
        config['batch_size'] = 2 ** config['batch_base2_exponent']
    batch_size = config['batch_size']
    display_step = compute_display_step(config, settings)
    session = get_tune_session()

    if run_mode == 'tune':
        if tune_restore == False:
            tune_dataframe = pd.DataFrame({
                'SI_dropout': [], 'SI_exp_kernel': [], 'SI_gen_fill': [], 'SI_gen_hidden_dim': [],
                'SI_gen_neck': [], 'SI_layer_norm': [], 'SI_normalize': [], 'SI_pad_mode': [],
                'batch_size': [], 'gen_lr': [], 'num_params': []
            })
            tune_dataframe.to_csv(tune_dataframe_path, index=False)
        else:
            tune_dataframe = pd.read_csv(tune_dataframe_path)

    if run_mode == 'test':
        test_dataframe = pd.DataFrame({
            'MSE (Network)': [], 'MSE (Recon1)': [], 'MSE (Recon2)': [],
            'SSIM (Network)': [], 'SSIM (Recon1)': [], 'SSIM (Recon2)': []
        })

    # Models: attenuation (frozen) and activity (trainable)
    # Create gen_atten first (no injection - only extracts features)
    atten_config = dict(config)
    atten_config['train_SI'] = True if flow_mode == 'coflow' else False
    gen_atten = create_generator(atten_config, device, gen_skip_handling='1x1Conv', enc_inject_channels=None, dec_inject_channels=None)
    
    # Extract channel tuples from gen_atten for gen_act injection
    enc_inject_ch = gen_atten.enc_stage_channels
    dec_inject_ch = gen_atten.dec_stage_channels
    
    # Create gen_act with injection parameters from gen_atten
    act_config = dict(config)
    act_config['train_SI'] = train_SI_act
    gen_act = create_generator(act_config, device, gen_skip_handling='1x1Conv', gen_flow_mode=flow_mode, enc_inject_channels=enc_inject_ch, dec_inject_channels=dec_inject_ch)

    # Loss for activity network (SI direction)
    base_criterion = config['sup_base_criterion']
    stats_criterion = config['SI_stats_criterion']
    alpha_min = config['SI_alpha_min']
    half_life_examples = config['SI_half_life_examples']
    hybrid_loss = HybridLoss(
        base_loss=base_criterion,
        stats_loss=stats_criterion,
        alpha_min=alpha_min,
        half_life_examples=half_life_examples,
    )

    # Path requirements: need both act and atten domains
    def require_path(key: str):
        if key not in paths or paths[key] is None:
            raise ValueError(f"Missing required path: '{key}'. Provide act_*/atten_* paths for frozen flow.")
    require_path('act_image_path')
    require_path('act_sino_path')
    require_path('atten_image_path')
    require_path('atten_sino_path')

    act_image_path = paths.get('act_image_path')
    act_sino_path = paths.get('act_sino_path')
    act_recon1_path = paths.get('act_recon1_path')
    act_recon2_path = paths.get('act_recon2_path')

    dataloader = DataLoader(
        NpArrayDataSet(
            act_sino_path=act_sino_path,
            act_image_path=act_image_path,
            config=config,
            settings=settings,
            augment=augment,
            offset=offset,
            num_examples=num_examples,
            sample_division=sample_division,
            device=device,
            act_recon1_path=act_recon1_path,
            act_recon2_path=act_recon2_path,
            atten_image_path=paths['atten_image_path'],
            atten_sino_path=paths['atten_sino_path'],
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        collate_fn=collate_nested,
    )

    # Checkpoints: tuning loads frozen atten only; training loads atten and act if present; save only act
    batch_step = 0
    gen_opt_state_dict = None
    if run_mode == 'tune':
        start_epoch, end_epoch, _, _, _ = init_checkpoint_state(False, run_mode, checkpoint_path + '-act', num_epochs)
        atten_ckpt = checkpoint_path + '-atten'
        if os.path.exists(atten_ckpt):
            gen_atten.load_state_dict(_load_frozen_state_dict(atten_ckpt, device))
        gen_atten.eval()
    else:
        start_epoch, end_epoch, batch_step, gen_state_dict, gen_opt_state_dict = init_checkpoint_state(
            load_state, run_mode, checkpoint_path + '-act', num_epochs
        )
        atten_ckpt = checkpoint_path + '-atten'
        if os.path.exists(atten_ckpt):
            gen_atten.load_state_dict(_load_frozen_state_dict(atten_ckpt, device))
        gen_atten.eval()
        if gen_state_dict:
            gen_act.load_state_dict(gen_state_dict)
        else:
            gen_act = gen_act.apply(weights_init_he)
    # Optimizer (activity only) will be created after feature scales are initialized
    gen_opt = None

    if run_mode in ('test', 'visualize'):
        gen_act.eval()

    mean_gen_loss = 0
    mean_CNN_SSIM = 0
    mean_CNN_MSE = 0
    report_num = 1

    time_init_full = time.time()
    time_init_loader = time.time()

    feat_scales = None

    for epoch in range(0 if run_mode in ('test', 'visualize') else start_epoch, 1 if run_mode in ('test', 'visualize') else end_epoch):
        for act_data, atten_data, recon_data in iter(dataloader):
            act_sino_scaled, act_image_scaled = act_data
            atten_sino_scaled, atten_image_scaled = atten_data
            act_recon1, act_recon2 = recon_data
            recon1_output = act_recon1
            recon2_output = act_recon2

            _ = display_times('loader time', time_init_loader, show_times)
            time_init_full = display_times('FULL STEP TIME', time_init_full, show_times)

            # Frozen forward
            with torch.no_grad():
                atten_input = atten_sino_scaled if flow_mode == 'coflow' else atten_image_scaled
                result = gen_atten(atten_input, return_features=True)
                frozen_enc_feats = result['encoder']
                frozen_dec_feats = result['decoder']

            # Initialize feature scales and optimizer once we have shapes
            if feat_scales is None:
                feat_scales = _build_feat_scale_dict(frozen_enc_feats, frozen_dec_feats, flow_mode, init_value=0.1)
                scale_params = list(feat_scales['encoder']) + list(feat_scales['decoder'])
                gen_opt = create_optimizer(gen_act, config)
                # add scale params group
                gen_opt.add_param_group({'params': scale_params, 'lr': config['gen_lr'], 'betas': (config['gen_b1'], config['gen_b2'])})
                if gen_opt_state_dict:
                    gen_opt.load_state_dict(gen_opt_state_dict)

            batch_tensors = {
                'act_sino_scaled': act_sino_scaled,
                'act_image_scaled': act_image_scaled,
                'atten_sino_scaled': atten_sino_scaled,
                'atten_image_scaled': atten_image_scaled,
            }
            # Activity input/target
            input_, target = route_batch_inputs(train_SI_act, batch_tensors, network_type='ACT')

            if run_mode in ('tune', 'train'):
                time_init_train = time.time()

                gen_opt.zero_grad()
                CNN_output = apply_feature_injection(
                    gen_act,
                    input_,
                    frozen_enc_feats,
                    frozen_dec_feats,
                    feat_scales,
                    flow_mode=flow_mode,
                    enable_inject_to_encoder=settings.get('enable_inject_to_encoder', True),
                    enable_inject_to_decoder=settings.get('enable_inject_to_decoder', True),
                )

                gen_loss = hybrid_loss(CNN_output, target)
                gen_loss.backward()
                gen_opt.step()

                if tune_debug:
                    log_tune_debug(gen_act, epoch, batch_step, gen_loss, device)

                mean_gen_loss += gen_loss.item() / display_step
                _ = display_times('training time', time_init_train, show_times)
            else:
                CNN_output = apply_feature_injection(
                    gen_act,
                    input_,
                    frozen_enc_feats,
                    frozen_dec_feats,
                    feat_scales if feat_scales is not None else _build_feat_scale_dict(frozen_enc_feats, frozen_dec_feats, flow_mode, init_value=0.1),
                    flow_mode=flow_mode,
                    enable_inject_to_encoder=settings.get('enable_inject_to_encoder', True),
                    enable_inject_to_decoder=settings.get('enable_inject_to_decoder', True),
                ).detach()

            if run_mode in ('tune', 'train'):
                mean_CNN_SSIM += calculate_metric(target, CNN_output, SSIM) / display_step
                mean_CNN_MSE += calculate_metric(target, CNN_output, MSE) / display_step

            if run_mode == 'test':
                test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM, recon1_output, recon2_output = \
                    compute_test_metrics('ACT', input_, CNN_output, target, act_image_scaled, test_dataframe, config, act_recon1, act_recon2)

            if run_mode == 'visualize':
                recon1_output, recon2_output = generate_reconstructions_for_visualization(recon1_output, recon2_output, input_, config)

            _ = display_times('metrics time', time.time(), show_times)

            if run_mode in ('tune', 'train') and (batch_step % display_step == 0):
                time_init_visualization = time.time()
                example_num = batch_step * batch_size

                batch_data = {
                    'input': input_,
                    'target': target,
                    'CNN_output': CNN_output,
                    'recon1_output': recon1_output,
                    'recon2_output': recon2_output,
                }

                if run_mode == 'tune' and session is not None:
                    tune_dataframe = report_tune_metrics(
                        gen_act, paths, config, settings, tune_dataframe, tune_dataframe_path,
                        train_SI_act, tune_dataframe_fraction, tune_max_t, report_num,
                        example_num, batch_step, epoch, session, device,
                    )
                    report_num += 1

                if run_mode == 'train':
                    visualize_train(batch_data, mean_gen_loss, mean_CNN_MSE, mean_CNN_SSIM, epoch, batch_step, example_num)

                mean_gen_loss = 0
                mean_CNN_SSIM = 0
                mean_CNN_MSE = 0

                _ = display_times('visualization time', time_init_visualization, show_times)

            time_init_loader = time.time()
            if run_mode not in ('test', 'visualize'):
                batch_step += 1

    if save_state and run_mode in ('train', 'tune'):
        checkpoint_dict = build_checkpoint_dict(gen_act, gen_opt, config, epoch + 1, batch_step)
        save_checkpoint(checkpoint_dict, checkpoint_path + '-act')

    if run_mode == 'test':
        return test_dataframe
*** End Patch