import os
import shutil
import ray
from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter, RunConfig, CheckpointConfig, FailureConfig
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.result_grid import ResultGrid

from FlexCNN_for_Medical_Physics.functions.main_run_functions.trainable import run_trainable
from FlexCNN_for_Medical_Physics.functions.main_run_functions.trainable_frozen_flow import run_trainable_frozen_flow

def tune_exp():
    print("placeholder")

def short_trial_dirname(trial):
    """Generate shorter trial directory names to avoid Windows 260-char path limit."""
    return f"trial_{trial.trial_id[-8:]}"  # Use last 8 chars of trial ID

def tune_networks(config, paths, settings, tune_opts, base_dirs):
    """
    Tune networks using Ray Tune.

    Parameters
    ----------
    config : dict
        Ray Tune search space dictionary (the parameter space to explore).
    paths : dict
        Dictionary of file paths (includes 'tune_storage_dirPath').
    settings : dict
        Runtime settings (run_mode, device, batch_size, etc.).
    tune_opts : dict
        Tuning-specific options with keys:
        'tune_metric', 'tune_minutes', 'tune_exp_name', 'tune_scheduler',
        'tune_restore', 'tune_max_t', 'tune_grace_period', 'num_CPUs', 'num_GPUs',
        'tune_search_alg' ('optuna' or 'hyperopt', default='optuna').
    base_dirs : dict
        Base directory paths (project_dirPath, etc.).

    Returns
    -------
    result_grid : ray.tune.ResultGrid
        Fitted tuning result grid.
    """
    # Force clean Ray restart to pick up new resource config
    try:
        ray.shutdown()
    except RuntimeError:
        # Ray was not initialized; proceed without error
        pass


    # Extract tune options
    tune_metric = tune_opts['tune_metric']
    tune_minutes = tune_opts['tune_minutes']
    tune_exp_name = tune_opts['tune_exp_name']
    tune_scheduler = tune_opts.get('tune_scheduler', 'ASHA')
    tune_restore = tune_opts.get('tune_restore')
    tune_max_t = tune_opts.get('tune_max_t')
    grace_period = tune_opts.get('tune_grace_period')
    num_CPUs = tune_opts.get('num_CPUs')
    num_GPUs = tune_opts.get('num_GPUs')
    cpus_per_trial = tune_opts.get('cpus_per_trial')
    gpus_per_trial = tune_opts.get('gpus_per_trial')
    tune_search_alg = tune_opts.get('tune_search_alg', 'optuna')  # 'optuna' or 'hyperopt'
    
    print(f"[DEBUG] num_GPUs={num_GPUs}, gpus_per_trial={gpus_per_trial}")

    os.environ.pop("AIR_VERBOSITY", None)

    # Disable Ray metrics exporter/dashboard to avoid RPC retry stalls in restricted envs
    os.environ["RAY_METRICS_EXPORT_PORT"] = "0"
    os.environ.setdefault("RAY_PROMETHEUS_MULTIPROC_DIR", os.path.join(base_dirs.get('project_dirPath', os.getcwd()), "ray_prometheus"))

    ray.init(ignore_reinit_error=True, num_cpus=num_CPUs, num_gpus=num_GPUs, include_dashboard=False)
    
    # Check what Ray actually detected
    resources = ray.available_resources()
    print(f"[DEBUG] Ray available resources: {resources}")

    # Extract tune_storage_dirPath directly from paths
    tune_storage_dirPath = paths['tune_storage_dirPath']

    ## What am I tuning for? ##
    if tune_metric == 'MSE':  # Values for these metric labels are passed to RayTune in the training function: session.report(.)
        optim_metric = 'MSE'
        min_max = 'min'  # minimise MSE
    elif tune_metric == 'SSIM':
        optim_metric = 'SSIM'
        min_max = 'max'  # maximize SSIM
    elif tune_metric == 'CUSTOM':
        optim_metric = 'CUSTOM'
        min_max = 'min'
    elif tune_metric == 'CR_symmetric':
        optim_metric = 'CR_symmetric'
        min_max = 'max'  # maximize hot lesion contrast recovery
    elif tune_metric == 'hot_underestimation':
        optim_metric = 'hot_underestimation'
        min_max = 'min'  # maximize cold lesion contrast recovery
    elif tune_metric == 'cold_overestimation':
        optim_metric = 'cold_overestimation'
        min_max = 'min'  # maximize weighted combination of hot/cold
    else:
        raise ValueError(f"Unsupported tune_metric='{tune_metric}'. Expected 'MSE', 'SSIM', 'CUSTOM', 'CR_symmetric', 'hot_underestimation', 'cold_overestimation'")


    print('===================')
    print('tune_max_t:', tune_max_t)
    print('optim_metric:', optim_metric)
    print('min_max:', min_max)
    print('grace_period:', grace_period)
    print('tune_minutes', tune_minutes)  # Set in "User Parameters".
    print('===================')

    ## Reporters ##
    # Simple CLIReporter - Ray will show parameters too, just zoom out or ignore them
    reporter = CLIReporter(
        metric_columns=['MSE', 'SSIM', 'CUSTOM', 'training_iteration', 'example_num'],
    )

    ## Trial Scheduler and Run Config ##
    if tune_scheduler == 'ASHA':
        scheduler = ASHAScheduler(
            time_attr='training_iteration',  # "Time" is measured in training iterations. 'training_iteration' is a RayTune keyword (not passed in session.report(...)).
            max_t=tune_max_t,  # (default=40). Maximum time units per trial (units = time_attr). Note: Ray Tune will by default run a maximum of 100 display steps (reports) per trial
            metric=optim_metric,  # This is the label in a dictionary passed to RayTune (in session.report(...))
            mode=min_max,
            grace_period=grace_period,  # Train for a minumum number of time_attr. Set in Tune() arguments.
            # reduction_factor=2
        )
        run_config = RunConfig(       # How to perform the run
            name=tune_exp_name,         # Ray checkpoints saved to this file, relative to tune_storage_dirPath. Set in "User Parameters"
            storage_path=tune_storage_dirPath,     # Tune search directory. Set in "User Parameters"
            progress_reporter=reporter,  # Specified above
            failure_config=FailureConfig(fail_fast=False),  # default = False. Keeps running if there is an error.
            checkpoint_config=CheckpointConfig(
                num_to_keep=10,         # Maximum number of checkpoints that are kept per run (for each trial)
                checkpoint_score_attribute=optim_metric,  # Determines which checkpoints are kept on disk.
                checkpoint_score_order=min_max
            )
        )
    else:
        scheduler = FIFOScheduler()     # First in/first out scheduler
        run_config = RunConfig(
            name=tune_exp_name,         # Ray checkpoints saved to this file, relative to tune_storage_dirPath
            storage_path=tune_storage_dirPath,     # Local directory
            progress_reporter=reporter,
            failure_config=FailureConfig(fail_fast=False),  # default = False
            checkpoint_config=CheckpointConfig(
                num_to_keep=10,         # Maximum number of checkpoints that are kept per run.
                checkpoint_score_attribute=optim_metric,  # Determines which checkpoints are kept on disk.
                checkpoint_score_order=min_max),
            stop={'training_iteration': tune_max_t},  # When using the FIFO scheduler, we must explicitly specify the stopping criterian.
        )

    ## Search Algorithm Selection ##
    # If the user has requested a fixed config for debugging (no tunable params), disable the searcher
    use_fixed_config = tune_opts.get('tune_force_fixed_config', tune_opts.get('use_fixed_config', False))
    if use_fixed_config:
        search_alg = None
    else:
        if tune_search_alg == 'hyperopt':
            search_alg = HyperOptSearch(metric=optim_metric, mode=min_max)
        elif tune_search_alg == 'optuna':
            search_alg = OptunaSearch(metric=optim_metric, mode=min_max)
        else:
            raise ValueError(f"tune_search_alg must be 'optuna' or 'hyperopt', got '{tune_search_alg}'")

    ## Unified trainable ##
    # Select appropriate trainable function based on network_type
    network_type = config.get('network_type')
    if network_type in ('FROZEN_COFLOW', 'FROZEN_COUNTERFLOW'):
        trainable_func = run_trainable_frozen_flow
    else:
        trainable_func = run_trainable
    
    trainable_param = tune.with_parameters(trainable_func, paths=paths, settings=settings)
    trainable_with_resources = tune.with_resources(trainable_param, {"CPU": cpus_per_trial, "GPU": gpus_per_trial})

    ## If starting from scratch ##
    if not tune_restore:
        # Check if search folder already exists
        tune_exp_path = os.path.join(tune_storage_dirPath, tune_exp_name)
        if os.path.exists(tune_exp_path):
            print(f"⚠️  [WARNING] Search folder already exists: {tune_exp_path}")
            print(f"⚠️  [WARNING] Ray Tune will append to existing results. To start fresh, manually delete this folder first.")

        # When debugging with a fixed config, run a single sample and don't use a searcher
        num_samples = 1 if use_fixed_config else -1

        # Initialize a blank tuner object
        tuner = tune.Tuner(
            trainable_with_resources,       # The objective function w/ resources
            param_space=config,             # Let RayTune know what parameter space (dictionary) to search over.
            tune_config=tune.TuneConfig(    # How to perform the search
                num_samples=num_samples,
                time_budget_s=tune_minutes * 60,  # time_budget is in seconds
                scheduler=scheduler,
                search_alg=search_alg,
                max_concurrent_trials=1,
                trial_dirname_creator=short_trial_dirname  # Shorter trial names to avoid path length issues
            ),
            run_config=run_config
        )

    ## If loading from a checkpoint ##
    else:
        # Load the tuner
        tuner = tune.Tuner.restore(
            path=os.path.join(tune_storage_dirPath, tune_exp_name),  # Path where previous run is checkpointed
            trainable=trainable_with_resources,
            resume_unfinished=False,
            resume_errored=False
        )

    result_grid: ResultGrid = tuner.fit()
    
    print("\n" + "="*80)
    print("[INFO] Tuning complete.")
    best_result = result_grid.get_best_result(metric=optim_metric, mode=min_max)
    print(f"[INFO] Best {optim_metric}: {best_result.metrics[optim_metric]:.6f}")
    print("="*80)
    return result_grid