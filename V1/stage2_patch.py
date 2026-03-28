"""
================================================================================
STAGE 2.5: ADVERSARIAL PATCHING (DOMAIN ADAPTATION)
Objective: Fix the "-800 AQI" panic response by fine-tuning on negative examples.
Status: PRODUCTION READY
================================================================================
"""
import os
import numpy as np
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

# INPUT: Your Best Model Weights (Fold 3)
MODEL_WEIGHTS = f"{BASE_DIR}/stage2_results/stage2_fold3_best.weights.h5"

# OUTPUT: Where to save the fixed model weights
OUTPUT_DIR = f"{BASE_DIR}/stage2_patched"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_WEIGHTS_PATH = f"{OUTPUT_DIR}/stage2_patched_model.weights.h5"

# THE PATCHING DATASET (Pseudo-Labels)
# (Path, Pseudo-Label AQI, Max Images to Load)
PATCH_CONFIG = [
    (f"{BASE_DIR}/Multi-class Weather Dataset/Rain", 40.0, 200),      # Rain = Clean-ish
    (f"{BASE_DIR}/blur/motion_blurred", 45.0, 100),                   # Blur = Ignore
    (f"{BASE_DIR}/clear_fog/foggy", 180.0, 150),                      # Fog = High but valid
    (f"{BASE_DIR}/Exposure_Errors_Dataset/INPUT_IMAGES", 20.0, 100),  # Glare = Clean
]

# Training Config
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4  # Low LR to prevent destroying good features

# ==========================================
# 2. DATA LOADER
# ==========================================
def load_patch_data():
    images = []
    labels = []
    
    print("🔄 Loading Patch Data...")
    total_found = 0
    
    for path, label, limit in PATCH_CONFIG:
        if not os.path.exists(path):
            print(f"❌ CRITICAL ERROR: Path not found: {path}")
            continue
            
        print(f"   📂 Scanning {os.path.basename(path)}... (Target: {label} AQI)")
        
        count = 0
        valid_files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for f in valid_files:
            if count >= limit: break
            try:
                img_path = os.path.join(path, f)
                img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img_arr = keras.preprocessing.image.img_to_array(img)
                
                images.append(img_arr)
                labels.append(label)
                count += 1
            except Exception as e:
                pass
        
        print(f"      -> Loaded {count} images.")
        total_found += count

    if total_found == 0:
        raise ValueError("❌ No images loaded! Check your paths.")
        
    print(f"✅ Total Patch Dataset: {len(images)} images.")
    
    # Convert to Tensor format
    X = preprocess_input(np.array(images))
    y = np.array(labels)
    return X, y

# ==========================================
# 3. MODEL BUILDER
# ==========================================
def build_and_load_model():
    print("🏗️  Rebuilding Model Architecture...")
    
    # 1. Define Input
    input_layer = layers.Input(shape=(224, 224, 3))
    
    # 2. Backbone (Frozen)
    backbone = keras.applications.EfficientNetB0(
        include_top=False, 
        weights=None, 
        input_tensor=input_layer
    )
    backbone.trainable = False  # FREEZE THE BACKBONE
    
    # 3. Rebuild Head
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
    
    # 4. Output
    output = layers.Dense(1, activation='linear')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    # 5. Load Weights
    print(f"🔄 Loading Weights from: {os.path.basename(MODEL_WEIGHTS)}")
    try:
        model.load_weights(MODEL_WEIGHTS)
        print("✅ Weights Loaded Successfully.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load weights! \n{e}")
        
    return model

# ==========================================
# 4. EXECUTION
# ==========================================
def main():
    # A. Load Data
    X_train, y_train = load_patch_data()
    
    # B. Load Model
    model = build_and_load_model()
    
    # C. Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mae',
        metrics=['mae']
    )
    
    # D. Train (Patching)
    print("\n🚀 STARTING ADVERSARIAL PATCHING...")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Backbone: FROZEN (Safe)")
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1
    )
    
    # E. Save
    print(f"\n💾 Saving Weights to: {OUTPUT_WEIGHTS_PATH}")
    model.save_weights(OUTPUT_WEIGHTS_PATH)
    
    print("\n" + "="*60)
    print(f"🏆 PATCH COMPLETE.")
    print("="*60)
    print("👉 NEXT STEP: Update 'diagnostic_stress_test.py' to use:")
    print(f"   MODEL_WEIGHTS = '{OUTPUT_WEIGHTS_PATH}'")

if __name__ == "__main__":
    main()