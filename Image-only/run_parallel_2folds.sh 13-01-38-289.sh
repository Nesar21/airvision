#!/bin/bash

# Run only TWO folds at a time (safe for Mac M4)

echo "Starting Fold 0 and Fold 1..."
python kfold_train_day.py --config cfg_kfold_day.yaml --fold 0 &
PID1=$!

python kfold_train_day.py --config cfg_kfold_day.yaml --fold 1 &
PID2=$!

wait $PID1
wait $PID2

echo "Fold 0 and Fold 1 done."

echo "Starting Fold 2 and Fold 3..."
python kfold_train_day.py --config cfg_kfold_day.yaml --fold 2 &
PID3=$!

python kfold_train_day.py --config cfg_kfold_day.yaml --fold 3 &
PID4=$!

wait $PID3
wait $PID4

echo "ALL FOLDS COMPLETED."
