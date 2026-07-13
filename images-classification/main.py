import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch.optim as optim
import optuna
import numpy as np
from codecarbon import EmissionsTracker
import argparse
import os

from models import get_model, MODELS
from datasets import get_dataloaders, DATASETS

def parse_args():

    parser = argparse.ArgumentParser(
        description="GreenAI HPO for Image Classification"
    )

    parser.add_argument("--multi_objective", type=int, default=1,
                        help="1 = multi-objective, 0 = single-objective")

    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--max_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return args

# =========================
# USO
# =========================

args = parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MULTI_OBJECTIVE = bool(args.multi_objective)
N_TRIALS = args.n_trials
MAX_TRIALS = args.max_trials
EPOCHS = args.epochs
SEED = args.seed
torch.manual_seed(SEED)

RESULT_DIR = os.path.expanduser("results")

os.makedirs(RESULT_DIR, exist_ok=True)

DATA_DIR = os.path.expanduser("datasets")
os.makedirs(DATA_DIR, exist_ok=True)

def safe_roc_auc(all_targets, all_probs, num_classes):
    try:
        if np.isnan(all_probs).any() or np.isinf(all_probs).any():
            print("WARNING: NaN/Inf detected in probabilities")
            return 0.0

        if np.isnan(all_targets).any() or np.isinf(all_targets).any():
            print("WARNING: NaN/Inf detected in targets")
            return 0.0

        unique_classes = np.unique(all_targets)

        if len(unique_classes) < 2:
            print("WARNING: Only one class present")
            return 0.0

        if num_classes == 2:
            probs = all_probs[:, 1]
            if np.isnan(probs).any():
                print("WARNING: NaN in binary probs")
                return 0.0

            return roc_auc_score(all_targets, probs)

        else:
            row_sums = all_probs.sum(axis=1)
            if np.any(row_sums == 0):
                print("WARNING: Zero probability rows")
                return 0.0

            all_probs = all_probs / row_sums[:, None]

            return roc_auc_score(
                all_targets,
                all_probs,
                multi_class="ovr"
            )

    except Exception as e:
        print(f"WARNING: AUC computation failed: {e}")
        return 0.0

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y = y.squeeze().long()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, num_classes):
    model.eval()
    correct, total = 0, 0
    all_probs, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y = y.squeeze().long()
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

    accuracy = correct / total
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    auc = safe_roc_auc(all_targets, all_probs, num_classes)
    return accuracy, auc


def build_objective(dataset, model_name, image_size, mode):

    def objective(trial):

        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        train_loader, test_loader, num_classes = get_dataloaders(dataset, batch_size, image_size, DATA_DIR)
        model = get_model(model_name, num_classes, dropout, DEVICE)

        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        criterion = nn.CrossEntropyLoss()

        tracker = EmissionsTracker(
            project_name="ml-images",
            experiment_id=f"{dataset}_{model_name}_trial{trial.number}",
            tracking_mode="process",
            output_dir=RESULT_DIR,
            log_level="error",
            output_file=(
                "ml-images_multi_emissions.csv"
                if MULTI_OBJECTIVE
                else "ml-images_single_emissions.csv"
            )
        )

        tracker.start()

        for _ in range(EPOCHS):
            train_one_epoch(model,train_loader,optimizer,criterion)

        acc, auc = evaluate(model, test_loader, num_classes)
        emissions = tracker.stop()

        trial.set_user_attr("emissions_kg",emissions)

        output_path = os.path.join(
            RESULT_DIR,
            (
                f"{dataset}_multi_results.json"
                if MULTI_OBJECTIVE
                else f"{dataset}_single_results.json"
            )
        )

        trial_result = {
            "trial": trial.number,
            "model": model_name,
            "image_size": image_size,
            "dataset": dataset,
            "seed": SEED,
            "accuracy": acc,
            "auc": auc,
            "emissions_kg": emissions,
            "params": trial.params
        }

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(trial_result)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Trial {trial.number} guardado")

        if MULTI_OBJECTIVE:
            return auc, emissions
        else:
            return auc

    return objective




if __name__ == "__main__":

    for dataset in DATASETS:
        for model_config in MODELS:

            model_name = model_config["model_name"]
            image_size = model_config["size"]

            print(f"\n=== Dataset: {dataset} | Model: {model_name} ===")
            mode = "multi" if MULTI_OBJECTIVE else "single"
            study_name = f"{dataset}_{model_name}_{mode}"
            storage = f"sqlite:///{RESULT_DIR}/optuna.db"

            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                directions=["maximize", "minimize"] if MULTI_OBJECTIVE else None,
                direction="maximize" if not MULTI_OBJECTIVE else None
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
                model_name=model_name,
                image_size=image_size,
                mode=mode
            )

            if trials_to_run > 0:
                study.optimize(objective, n_trials=trials_to_run)

            else:
                print("Nada que ejecutar.")