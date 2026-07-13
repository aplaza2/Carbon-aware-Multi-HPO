import optuna
import torch
import os
import json
from recbole.quick_start import run_recbole
from codecarbon import EmissionsTracker
from config import BASE_CONFIG, DATASETS_CONFIG
from utils import parse_args, build_recommender_config

_original_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)

torch.load = _patched_load


args = parse_args()

dataset = args.dataset
recommender = args.recommender
MODE = args.mode
N_TRIALS = args.trials
SEED = args.seed
METRIC = args.metric
MAX_TRIALS = args.maxtrials

MULTI_OBJECTIVE = (MODE == "multi")

print("Dataset:", dataset)
print("Recommender:", recommender)
print("Mode:", MODE)
print("Trials:", N_TRIALS)
print("Seed:", SEED)
print("Metric:", METRIC)

result_dir = os.path.expanduser("results/")

os.makedirs(result_dir, exist_ok=True)

dataset_config = next(
    (ds for ds in DATASETS_CONFIG if ds["name"] == dataset),
    None
)

if dataset_config is None:
    raise ValueError(f"Dataset '{dataset}' not found in DATASETS_CONFIG")

DATASET_PARAMS = {
    'USER_ID_FIELD': dataset_config["session_key"],
    'ITEM_ID_FIELD': dataset_config["item_key"],
    'TIME_FIELD': dataset_config["time_key"],
    'load_col': {
        'inter': [
            dataset_config["session_key"],
            dataset_config["item_key"],
            dataset_config["time_key"]
        ]
    },
}

def objective(trial):

    config_dict = {
        **BASE_CONFIG,
        **DATASET_PARAMS,
        "seed": SEED
    }

    config_dict.update(
        build_recommender_config(trial, recommender)
    )

    tracker = EmissionsTracker(
        project_name="optuna_base_recommenders",
        experiment_id=f"{dataset}_{recommender}_trial{trial.number}",
        tracking_mode="process",
        log_level="error",
        output_dir=result_dir,
        output_file=f"{dataset}_{MODE}_emissions.csv"
    )

    tracker.start()

    results = run_recbole(
        model=recommender,
        dataset=dataset,
        config_dict=config_dict,
        saved=False
    )

    emissions = tracker.stop()

    score = results["best_valid_result"][METRIC]

    trial.set_user_attr("emissions_kg", emissions)

    output_path = os.path.join(
        result_dir,
        f"{dataset}_{MODE}_results.json"
    )

    trial_result = {
        "trial": trial.number,
        "recommender": recommender,
        "rl_strategy": None,
        "dataset": dataset,
        "seed": SEED,
        "emissions_kg": emissions,
        "results": results,
        "params": trial.params
    }

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(trial_result)

    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Trial {trial.number} saved")

    if MULTI_OBJECTIVE:
        return score, emissions
    else:
        return score

study_name = f"{dataset}_{recommender}_{MODE}"

study = optuna.create_study(
    study_name=study_name,
    storage=f"sqlite:///{result_dir}/optuna.db",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(),
    directions=["maximize", "minimize"] if MULTI_OBJECTIVE else None,
    direction="maximize" if not MULTI_OBJECTIVE else None
)

completed_trials = sum(
    1 for t in study.trials
    if t.state == optuna.trial.TrialState.COMPLETE
)

if MAX_TRIALS is not None:
    remaining_trials = MAX_TRIALS - completed_trials
    if remaining_trials <= 0:
        print(f"Completed trials: {completed_trials}")
        trials_to_run = 0
    else:
        trials_to_run = min(N_TRIALS, remaining_trials)
else:
    trials_to_run = N_TRIALS

if trials_to_run > 0:
    print(f"Running {trials_to_run} additional trials...")
    study.optimize(objective, n_trials=trials_to_run)
else:
    print("No additional trials to run.")


if MULTI_OBJECTIVE:
    print("\n Pareto front trials:")
    for t in study.best_trials:
        print(
            f"Trial {t.number} → "
            f"{METRIC}: {t.values[0]} | "
            f"CO2e: {t.values[1]}"
        )

else:

    best = study.best_trial

    print("\nBest trial:")
    print(
        f"Trial {best.number} → "
        f"{METRIC}: {best.value} | "
        f"CO2e: {best.user_attrs.get('emissions_kg')}"
    )