import argparse

def build_recommender_config(trial, recommender):

    config = {}

    config["learner"] = trial.suggest_categorical(
        "rec_learner",
        ["adam", "sgd", "adagrad", "rmsprop"]
    )

    config["train_batch_size"] = trial.suggest_categorical(
        "rec_train_batch_size",
        [512, 1024, 2048]
    )

    config["learning_rate"] = trial.suggest_categorical(
        "learning_rate",
        [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    )

    loss_type = trial.suggest_categorical(
        "rec_loss_type",
        ["CE", "BPR"]
    )

    config["loss_type"] = loss_type

    if loss_type == "BPR":
        config["train_neg_sample_args"] = {
            "distribution": "uniform",
            "sample_num": 1
        }
    else:
        config["train_neg_sample_args"] = None

    if recommender in ["NextItNet", "Caser"]:
        config["embedding_size"] = trial.suggest_categorical(
            "rec_embedding_size",
            [32, 64, 128, 256]
        )
    else:
        config["hidden_size"] = trial.suggest_categorical(
            "rec_hidden_size",
            [32, 64, 128, 256]
        )

    return config


def parse_args(with_rl=False):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["single", "multi"],
        help="Optimization mode"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ml-100k", "diginetica", "lastfm", "retailrocket", "yoochoose"],
        help="Dataset name"
    )

    parser.add_argument(
        "--recommender",
        type=str,
        required=True,
        choices=["GRU4Rec", "SASRec", "RepeatNet", "NextItNet", "Caser"],
        help="Backbone recommender"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="ndcg@10",
        choices=["ndcg@10", "mrr@10", "recall@10", "hit@10", "precision@10"],
        help="Evaluation metric"
    )

    parser.add_argument(
        "--maxtrials",
        type=int,
        default=None,
        help="Max number of trials allowed for the study"
    )

    return parser.parse_args()