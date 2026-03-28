import os, pandas as pd, torch
from pathlib import Path

print("\n=== VERIFY DAY-ONLY K-FOLD ===")

root = Path(".")

# MASTER
m = root/"data/master_v3_day_only.csv"
print("\nMaster:", m, "OK" if m.exists() else "MISSING")
df = pd.read_csv(m)
print(" rows:", len(df), "TRAQID:", (df['source']=="TRAQID").sum())

# SPLITS
sd = root/"splits_dayonly"
print("\nSplits dir:", sd, "OK" if sd.exists() else "MISSING")

for f in range(4):
    fd = sd/f"fold{f}"
    print(f"\nFold {f}:", "OK" if fd.exists() else "MISSING")
    for n in ["train.csv","val.csv","test.csv"]:
        p = fd/n
        print(" ",n,":","OK" if p.exists() else "MISSING")

# IMAGE sample
print("\nSample image check:")
imgp = df["image_path"].iloc[0]
print(" ", imgp, "OK" if os.path.exists(imgp) else "MISSING")

# DEPTH sample
if "depth_path" in df.columns:
    dp = df["depth_path"].dropna().iloc[0]
    print(" depth_path:", dp, "OK" if os.path.exists(dp) else "MISSING")

# CONFIG
cfg = root/"cfg_kfold_day.yaml"
print("\nConfig:", cfg, "OK" if cfg.exists() else "MISSING")

# SCRIPTS
for s in ["kfold_train_day.py", "litefusion_model_and_train.py"]:
    print(" ", s, ":","OK" if (root/s).exists() else "MISSING")

# MPS
print("\nMPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

print("\n=== DONE ===\n")
