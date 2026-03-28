"""
================================================================================
GENERATOR: PATCHED VISION PREDICTIONS (HARDENED v2)
Objective: Generate clean AQI predictions for the entire dataset using
           the stabilized Stage 2.5 model. These inputs feed Stage 3.
Status: PRODUCTION READY (Atomic Row Alignment Fixed)
================================================================================
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# 0. REPRODUCIBILITY LOCK
# ==========================================
tf.random.set_seed(42)
np.random.seed(42)

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = "/Users/nesar/VS/V1"
DATA_CSV = f"{BASE_DIR}/master_stage2_data.csv"
MODEL_WEIGHTS = f"{BASE_DIR}/stage2_patched/stage2_patched_model.weights.h5"
OUTPUT_CSV = f"{BASE_DIR}/stage2_patched_predictions.csv"
BATCH_SIZE = 64  # Faster inference

# ==========================================
# 2. DATA GENERATOR (SAFE)
# ==========================================
class InferenceGenerator(keras.utils.Sequence):
    def __init__(self, df, batch_size=32):
        self.df = df
        self.batch_size = batch_size
        self.indices = np.arange(len(df))

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_rows = self.df.iloc[batch_indices]
        
        images = []
        # Paths are pre-validated in main(), so no complex logic needed here
        for _, row in batch_rows.iterrows():
            img_path = row['valid_path'] 
            
            img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_arr = keras.preprocessing.image.img_to_array(img)
            images.append(img_arr)
        
        return preprocess_input(np.array(images))

# ==========================================
# 3. LOAD MODEL
# ==========================================
def build_model():
    print("🏗️  Rebuilding Model...")
    input_layer = layers.Input(shape=(224, 224, 3))
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_tensor=input_layer
    )
    
    x = backbone.output
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    concat = layers.Concatenate()([gap, gmp])
    
    x = layers.Dense(256, activation='relu')(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(1, activation='linear')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    print(f"🔄 Loading Patched Weights: {os.path.basename(MODEL_WEIGHTS)}")
    try:
        model.load_weights(MODEL_WEIGHTS)
        print("✅ Weights Loaded Successfully.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load weights: {e}")
        
    return model

# ==========================================
# 4. EXECUTION
# ==========================================
def main():
    # A. Validate Input CSV
    if not os.path.exists(DATA_CSV):
        raise ValueError(f"❌ Master CSV not found: {DATA_CSV}")
        
    df = pd.read_csv(DATA_CSV)
    print(f"📂 Loaded Raw List: {len(df)} images")
    
    # Check Required Columns
    required = {'image_path', 'AQI', 'dataset'}
    if not required.issubset(df.columns):
        raise ValueError(f"❌ CSV missing columns! Found: {df.columns}")

    # B. Pre-Validate Paths (ATOMIC FIX)
    print("🔍 Pre-validating image paths...")
    valid_rows = []
    
    # Iterate once, check existence, and build a clean list of safe rows
    for _, row in df.iterrows():
        path = row['image_path']
        # Handle relative/absolute logic safely
        full_path = path if os.path.exists(path) else os.path.join(BASE_DIR, path)
        
        if os.path.exists(full_path):
            # Convert row to dict to preserve data
            row_dict = row.to_dict()
            row_dict['valid_path'] = full_path
            valid_rows.append(row_dict)
            
    if not valid_rows:
        raise ValueError("❌ No valid images found! Check paths in CSV.")

    # Rebuild DataFrame from clean list (Guarantees alignment)
    df = pd.DataFrame(valid_rows)
    print(f"✅ Final Valid Dataset: {len(df)} images")

    # C. Initialize Generator
    test_gen = InferenceGenerator(df, batch_size=BATCH_SIZE)
    
    # D. Load Model & Predict
    model = build_model()
    
    print(f"\n🚀 Generating Predictions for {len(df)} images...")
    preds = model.predict(test_gen, verbose=1)
    
    # E. Assertions (Strict Hygiene)
    # Trim potential padding from the generator
    preds = preds[:len(df)]
    
    if len(preds) != len(df):
        raise AssertionError(f"❌ Prediction mismatch! DF: {len(df)}, Preds: {len(preds)}")
    
    # F. Save Results
    df['stage2_patched_pred'] = preds.flatten()
    
    # Export clean columns required for Stage 3
    # We include 'image_path' to merge with Physics features later if needed
    output_cols = ['image_path', 'AQI', 'stage2_patched_pred', 'dataset']
    df[output_cols].to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*60)
    print(f"🏆 DATA READY.")
    print(f"💾 Saved {len(df)} clean predictions to: {OUTPUT_CSV}")
    print("="*60)
    print("👉 NEXT STEP: Run Stage 3 Training Script")

if __name__ == "__main__":
    main()