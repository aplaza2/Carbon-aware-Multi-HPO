import os
import subprocess
import argparse

ALGOS = [
    "ppo",
    "a2c",
    "dqn",
    "ars",
]

ENVS = [
    "BreakoutNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "AsteroidsNoFrameskip-v4",
]

RESULTS_DIR = "results"


def run_optuna(algo, env, mode, n_trials):
    study_name = f"{env}_{algo}_{mode}"
    db_path = os.path.join(RESULTS_DIR, "optuna.db")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("================================")
    print(f"Env={env}")
    print(f"Algo={algo}")
    print(f"Mode={mode}")
    print("================================")

    cmd = [
        "python",
        "train.py",
        "--algo", algo,
        "--env", env,
        "--optimize",
        "--storage", f"sqlite:///{db_path}",
        "--study-name", study_name,
        "--n-trials", n_trials,
        "-n", "1000000",
        "--mode", mode,
        "--device", "cuda",
        "--verbose", 0,
        "--optuna-results-path", RESULTS_DIR,
        "--max-total-trials", str(20)
    ]

    subprocess.run(cmd, check=True)

    print(f"<<< Finished {algo} on {env}\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--start-env", default="")
    parser.add_argument("--start-algo", default="")
    parser.add_argument("--mode", default="single")
    parser.add_argument("--n-trials", default=5, type=int)

    args = parser.parse_args()

    skip_envs = bool(args.start_env)

    for env in ENVS:

        if skip_envs:
            if env != args.start_env:
                continue
            skip_envs = False

        print("================================")
        print(f"Environment: {env}")
        print("================================")

        skip_algos = bool(args.start_algo)

        for algo in ALGOS:

            if skip_algos:
                if algo != args.start_algo:
                    continue
                skip_algos = False

            run_optuna(algo, env, args.mode, args.n_trials)

    print("================================")
    print("All Optuna experiments finished")
    print("================================")


if __name__ == "__main__":
    main()