# run_all_folds.py
# Run all 4 folds sequentially.
#
# Usage:
#   python run_all_folds.py --config cfg.yaml
#
# This will run:
#   python litefusion_kfold_train.py --fold 0 --config cfg.yaml
#   python litefusion_kfold_train.py --fold 1 --config cfg.yaml
#   python litefusion_kfold_train.py --fold 2 --config cfg.yaml
#   python litefusion_kfold_train.py --fold 3 --config cfg.yaml
#
# Results saved in:
#   outputs/fold0/best.pth
#   outputs/fold1/best.pth
#   outputs/fold2/best.pth
#   outputs/fold3/best.pth

import json
import argparse
import subprocess
from pathlib import Path

def run_all_folds(cfg_path):
    folds = [0,1,2,3]
    results = {}

    for f in folds:
        print("="*80)
        print(f"STARTING FOLD {f}")
        print("="*80)

        cmd = [
            "python", "litefusion_kfold_train.py",
            "--fold", str(f),
            "--config", cfg_path
        ]

        # launch training
        proc = subprocess.run(cmd, capture_output=False)

        if proc.returncode != 0:
            print(f"FOLD {f} FAILED")
            break

        # read score from the output logs? No — simpler:
        # load best MAE from each fold.json file if needed.
        fold_result_file = Path("outputs")/f"fold{f}"/"best.pth"
        results[f] = str(fold_result_file)

        print(f"FINISHED FOLD {f}")

    print("="*80)
    print("ALL FOLDS COMPLETED")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    run_all_folds(args.config)
