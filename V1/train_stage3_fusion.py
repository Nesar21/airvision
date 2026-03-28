"""
================================================================================
STAGE 3: PHYSICS–VISION FUSION (ROBUST DATA LOADER)
Objective:
- Fuse Stage 1 (Physics features) and Stage 2.5 (Vision predictions)
- Handle CSV column mismatch automatically
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================================================================
# 0. REPRODUCIBILITY
# ==============================================================================
tf.random.set_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. CONFIG
# ==============================================================================
BASE_DIR = "/Users/nesar/VS/V1"
OUTPUT_DIR = f"{BASE_DIR}/stage3_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHYSICS_CSV = f"{BASE_DIR}/features/model4_physics_features.csv"
VISION_CSV  = f"{BASE_DIR}/stage2_patched_predictions.csv"

# Expected Physics Columns (we will try to match these)
PHYSICS_COLS = [
    'contrast', 'saturation', 'brightness', 'dark_channel',
    'transmission', 'atmos_light', 'edge_density', 'color_temp'
]

# ==============================================================================
# 2. LOAD & ALIGN DATA (ROBUST VERSION)
# ==============================================================================
def load_data():
    if not os.path.exists(PHYSICS_CSV):
        raise FileNotFoundError(f"Missing: {PHYSICS_CSV}")
    if not os.path.exists(VISION_CSV):
        raise FileNotFoundError(f"Missing: {VISION_CSV}")

    # 1. Load Physics
    print(f"Loading Physics CSV: {os.path.basename(PHYSICS_CSV)}")
    df_phy = pd.read_csv(PHYSICS_CSV)
    
    # 🔧 FIX: Normalize column names (strip spaces, lowercase)
    df_phy.columns = df_phy.columns.str.strip().str.lower()
    
    # 🕵️ CHECK: Do we have the columns?
    missing = [c for c in PHYSICS_COLS if c not in df_phy.columns]
    if missing:
        print("\n❌ CRITICAL ERROR: Physics CSV is missing expected columns!")
        print(f"   Missing: {missing}")
        print(f"   Found Columns: {list(df_phy.columns)}")
        print("   -> Please update PHYSICS_COLS in the script to match 'Found Columns'.")
        sys.exit(1)

    # 2. Load Vision
    print(f"Loading Vision CSV: {os.path.basename(VISION_CSV)}")
    df_vis = pd.read_csv(VISION_CSV)
    df_vis.columns = df_vis.columns.str.strip().str.lower() # Normalize vision too

    # 3. Merge
    # We align on 'image_path'. Ensure it exists in both.
    if 'image_path' not in df_phy.columns:
        raise KeyError(f"Physics CSV missing 'image_path'. Found: {list(df_phy.columns)}")
    if 'image_path' not in df_vis.columns:
        raise KeyError(f"Vision CSV missing 'image_path'. Found: {list(df_vis.columns)}")

    print("🔗 Merging Datasets...")
    # Select only necessary columns to avoid duplicates
    df = pd.merge(
        df_vis[['image_path', 'stage2_patched_pred', 'aqi']], # 'aqi' normalized from 'AQI'
        df_phy[['image_path'] + PHYSICS_COLS],
        on='image_path',
        how='inner'
    )

    if len(df) == 0:
        raise ValueError("❌ Merged dataset is empty! Check if 'image_path' values match in both CSVs.")

    print(f"✅ Aligned Dataset: {len(df)} samples")
    return df

# ==============================================================================
# 3. MODEL
# ==============================================================================
def build_fusion_model(n_phy):
    # Inputs
    phy_in = layers.Input(shape=(n_phy,), name="physics_input")
    vis_in = layers.Input(shape=(1,), name="vision_input")

    # Physics branch
    p = layers.Dense(32, activation="relu")(phy_in)
    p = layers.Dense(16, activation="relu")(p)
    phy_pred = layers.Dense(1, name="aux_physics_output")(p)

    # Vision calibration
    v = layers.Dense(8, activation="relu")(vis_in)

    # Gate
    gate_x = layers.Concatenate()([p, v, phy_in])
    gate_x = layers.Dense(16, activation="relu")(gate_x)
    gate = layers.Dense(1, activation="sigmoid", name="gate_weight")(gate_x)

    # Fusion
    fused = layers.Add()([
        layers.Multiply()([gate, phy_pred]),
        layers.Multiply()([1.0 - gate, vis_in])
    ])

    final_out = layers.ReLU(max_value=500.0, name="final_output")(fused)

    model = keras.Model(
        inputs=[phy_in, vis_in],
        outputs=[final_out, phy_pred]
    )

    return model

# ==============================================================================
# 4. TRAIN & EVALUATE
# ==============================================================================
def main():
    df = load_data()

    X_phy = df[PHYSICS_COLS].values.astype("float32")
    X_vis = df["stage2_patched_pred"].values.astype("float32").reshape(-1, 1)
    y     = df["aqi"].values.astype("float32") # Normalized to lowercase 'aqi'

    Xp_tr, Xp_te, Xv_tr, Xv_te, y_tr, y_te = train_test_split(
        X_phy, X_vis, y, test_size=0.2, random_state=42
    )

    model = build_fusion_model(len(PHYSICS_COLS))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            "final_output": "mae",
            "aux_physics_output": "mae",
        },
        loss_weights={
            "final_output": 1.0,
            "aux_physics_output": 0.3,
        },
        metrics={
            "final_output": ["mae"],
        }
    )

    print("🚀 Training Stage 3 Fusion Model...")
    model.fit(
        [Xp_tr, Xv_tr],
        {"final_output": y_tr, "aux_physics_output": y_tr},
        validation_data=(
            [Xp_te, Xv_te],
            {"final_output": y_te, "aux_physics_output": y_te},
        ),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # Evaluate
    final_preds, phy_preds = model.predict([Xp_te, Xv_te])
    final_preds = final_preds.flatten()
    phy_preds   = phy_preds.flatten()

    mae = mean_absolute_error(y_te, final_preds)
    r2  = r2_score(y_te, final_preds)
    phy_mae = mean_absolute_error(y_te, phy_preds)

    # Gate inspection
    gate_model = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("gate_weight").output
    )
    gate_vals = gate_model.predict([Xp_te, Xv_te]).flatten()

    print("\n" + "="*60)
    print("🏆 STAGE 3 RESULTS")
    print(f"Final MAE:        {mae:.2f}")
    print(f"Final R²:         {r2:.3f}")
    print(f"Physics-only MAE: {phy_mae:.2f}")
    print("-"*30)
    print(f"Gate Avg: {gate_vals.mean():.4f}")
    
    if gate_vals.mean() < 0.05:
        print("⚠️ Gate collapsed to Vision")
    elif gate_vals.mean() > 0.95:
        print("⚠️ Gate collapsed to Physics")
    else:
        print("✅ Gate is actively arbitrating")

    print("="*60)

    # Save
    model.save(f"{OUTPUT_DIR}/stage3_fusion_model.h5")

    pd.DataFrame({
        "True_AQI": y_te,
        "Pred_AQI": final_preds,
        "Physics_Pred": phy_preds,
        "Gate": gate_vals,
        "Vision_Input": Xv_te.flatten()
    }).to_csv(f"{OUTPUT_DIR}/stage3_test_results.csv", index=False)

if __name__ == "__main__":
    main()