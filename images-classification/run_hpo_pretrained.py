import os
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from codecarbon import EmissionsTracker
from transformers import AutoImageProcessor
from main import train_one_epoch, evaluate
from datasets import get_dataloaders, DATASETS
from models_pretrained import get_pretrained_model, PRETRAINED_MODELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_objective", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--max_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


args = parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MULTI_OBJECTIVE = bool(args.multi_objective)
N_TRIALS = args.n_trials
MAX_TRIALS = args.max_trials
EPOCHS = args.epochs
SEED = args.seed

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

RESULT_DIR = os.path.expanduser("~/archive/ml-images_pretrained_results")
os.makedirs(RESULT_DIR, exist_ok=True)

DATA_DIR = os.path.expanduser("~/archive/datasets")
os.makedirs(DATA_DIR, exist_ok=True)

def get_processor_stats(hf_id):
    processor = AutoImageProcessor.from_pretrained(hf_id)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    return mean, std


def build_optimizer(model, optimizer_name, lr_head, lr_backbone, weight_decay):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("classifier"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []

    if len(backbone_params) > 0:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            }
        )

    if len(head_params) > 0:
        param_groups.append(
            {
                "params": head_params,
                "lr": lr_head,
                "weight_decay": weight_decay,
            }
        )

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(param_groups)

    elif optimizer_name == "adam":
        optimizer = optim.Adam(param_groups)

    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            param_groups,
            momentum=0.9,
        )

    else:
        raise ValueError(f"Optimizer no soportado: {optimizer_name}")

    return optimizer


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def build_objective(dataset, model_config, mode):
    model_name = model_config["model_name"]
    family = model_config["family"]
    hf_id = model_config["hf_id"]
    image_size = model_config["size"]

    mean, std = get_processor_stats(hf_id)

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        finetune_strategy = trial.suggest_categorical(
            "finetune_strategy",
            ["linear_probe", "partial_finetune", "full_finetune"]
        )

        if finetune_strategy == "linear_probe":
            unfreeze_last_n = 0

        elif finetune_strategy == "partial_finetune":
            unfreeze_last_n = trial.suggest_categorical(
                "unfreeze_last_n",
                [1, 2, 4]
            )

        else:
            unfreeze_last_n = 999

        lr_head = trial.suggest_float("lr_head", 1e-5, 1e-3, log=True)

        if finetune_strategy == "linear_probe":
            lr_backbone = 0.0
        else:
            lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 5e-5, log=True)

        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        optimizer_name = trial.suggest_categorical(
            "optimizer",
            ["adamw", "adam"]
        )
        
        trial_epochs = EPOCHS

        train_loader, test_loader, num_classes = get_dataloaders(
            dataset,
            batch_size,
            image_size,
            DATA_DIR,
            normalize=True
        )

        model = get_pretrained_model(
            family=family,
            hf_id=hf_id,
            num_classes=num_classes,
            dropout=dropout,
            finetune_strategy=finetune_strategy,
            unfreeze_last_n=unfreeze_last_n,
            device=DEVICE,
        )

        optimizer = build_optimizer(
            model=model,
            optimizer_name=optimizer_name,
            lr_head=lr_head,
            lr_backbone=lr_backbone,
            weight_decay=weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        trainable_params = count_trainable_params(model)
        total_params = count_total_params(model)

        tracker = EmissionsTracker(
            project_name="ml-images-vit",
            experiment_id=f"{dataset}_{model_name}_trial{trial.number}",
            tracking_mode="process",
            output_dir=RESULT_DIR,
            log_level="error",
            output_file=(
                "ml-images_multi_emissions.csv"
                if MULTI_OBJECTIVE
                else "ml-images_single_emissions.csv"
            ),
        )

        tracker.start()

        for _ in range(trial_epochs):
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
            )

        acc, auc = evaluate(
            model=model,
            loader=test_loader,
            num_classes=num_classes,
        )

        emissions = tracker.stop()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trial.set_user_attr("emissions_kg", emissions)
        trial.set_user_attr("accuracy", acc)
        trial.set_user_attr("auc", auc)
        trial.set_user_attr("trainable_params", trainable_params)
        trial.set_user_attr("total_params", total_params)

        output_path = os.path.join(
            RESULT_DIR,
            (
                f"{dataset}_pretrained_multi_results.json"
                if MULTI_OBJECTIVE
                else f"{dataset}_pretrained_single_results.json"
            )
        )

        trial_result = {
            "trial": trial.number,
            "model": model_name,
            "family": family,
            "hf_id": hf_id,
            "image_size": image_size,
            "dataset": dataset,
            "seed": SEED,
            "accuracy": acc,
            "auc": auc,
            "emissions_kg": emissions,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "params": {
                **trial.params,
                "resolved_unfreeze_last_n": unfreeze_last_n,
                "trial_epochs": trial_epochs,
            },
        }

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(trial_result)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

        print(
            f"Trial {trial.number} guardado | "
            f"model={model_name} | "
            f"strategy={finetune_strategy} | "
            f"auc={auc:.4f} | "
            f"emissions={emissions:.6f} kg"
        )

        if MULTI_OBJECTIVE:
            return auc, emissions

        return auc

    return objective


if __name__ == "__main__":
    for dataset in DATASETS:
        for model_config in PRETRAINED_MODELS:
            model_name = model_config["model_name"]

            print(f"\n=== Dataset: {dataset} | Pretrained model: {model_name} ===")

            mode = "multi" if MULTI_OBJECTIVE else "single"
            study_name = f"{dataset}_{model_name}_pretrained_{mode}"

            storage_path = os.path.join(RESULT_DIR, "optuna_pretrained.db")
            storage = f"sqlite:///{storage_path}"

            if MULTI_OBJECTIVE:
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage,
                    load_if_exists=True,
                    directions=["maximize", "minimize"],
                )
            else:
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage,
                    load_if_exists=True,
                    direction="maximize",
                )

            completed_trials = sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            )

            if MAX_TRIALS is not None:
                remaining = MAX_TRIALS - completed_trials

                if remaining <= 0:
                    print("MAX_TRIALS alcanzado")
                    continue

                trials_to_run = min(N_TRIALS, remaining)

            else:
                trials_to_run = N_TRIALS

            print(f"Running {trials_to_run} trials...")

            objective = build_objective(
                dataset=dataset,
                model_config=model_config,
                mode=mode,
            )

            if trials_to_run > 0:
                study.optimize(
                    objective,
                    n_trials=trials_to_run,
                )
            else:
                print("Nada que ejecutar.")