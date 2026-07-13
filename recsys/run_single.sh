DATASETS=("lastfm" "diginetica" "retailrocket" "yoochoose")
RECOMMENDERS=("GRU4Rec" "RepeatNet" "SASRec" "Caser" "NextItNet")

TRIALS=20
SEED=42
MODE="single"
METRIC="ndcg@10"
MAX_TRIALS=20

for dataset in "${DATASETS[@]}"
do
  for recommender in "${RECOMMENDERS[@]}"
  do
    echo "========================================"
    echo "Dataset: $dataset | Model: $recommender"
    echo "========================================"

    python main_optuna_base.py \
      --dataset $dataset \
      --recommender $recommender \
      --mode $MODE \
      --trials $TRIALS \
      --seed $SEED \
      --metric $METRIC \
      --maxtrials $MAX_TRIALS

  done
done