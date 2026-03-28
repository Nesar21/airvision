"""
================================================================================
FORENSIC STRESS TEST: STAGE 2 VISION MODEL
Author: Nesar (The Architect)
Objective: Quantify failure rates on confounding variables (Rain, Blur, Glare).
================================================================================
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = "/Users/nesar/VS/V1"
# We use Fold 3 because it had the best validation metrics
MODEL_WEIGHTS = f"{BASE_DIR}/stage2_patched/stage2_patched_model.weights.h5"

# The "Enemy" Datasets to Audit
STRESS_ZONES = {
    "RAIN (Should be Clean)": {
        "path": f"{BASE_DIR}/Multi-class Weather Dataset/Rain",
        "expected": "LOW (<60)",
        "fail_threshold": 100  # If > 100, model thinks Rain = Smog
    },
    "MOTION BLUR (Should be Ignored)": {
        "path": f"{BASE_DIR}/blur/motion_blurred",
        "expected": "LOW/MODERATE",
        "fail_threshold": 150 # If > 150, model thinks Blur = Smog
    },
    "NIGHT (Should be Variable)": {
        "path": f"{BASE_DIR}/day_night_images/test/night",
        "expected": "UNKNOWN",
        "fail_threshold": 300 # If > 300, model confuses ISO noise for PM2.5
    },
    "EXPOSURE ERROR (Glare)": {
        "path": f"{BASE_DIR}/Exposure_Errors_Dataset/INPUT_IMAGES",
        "expected": "LOW (Blinded)",
        "fail_threshold": -10 # Check for negative values
    },
    "REAL FOG (The Confuser)": {
        "path": f"{BASE_DIR}/clear_fog/foggy",
        "expected": "HIGH",
        "fail_threshold": 0 # Just for logging
    }
}

# ==========================================
# 2. MODEL ARCHITECTURE (Must match Training)
# ==========================================
def build_model():
    input_layer = layers.Input(shape=(224, 224, 3), name='input')
    
    # Rebuild backbone structure
    backbone = EfficientNetB0(
        include_top=False,
        weights=None, # We load custom weights later
        input_tensor=input_layer,
        pooling=None
    )
    
    x = backbone.output
    gap = layers.GlobalAveragePooling2D(name='gap')(x)
    gmp = layers.GlobalMaxPooling2D(name='gmp')(x)
    concat = layers.Concatenate(name='concat')([gap, gmp])
    
    # Dense Head
    x = layers.Dense(256, activation='relu', name='dense1')(concat)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.3, name='dropout1')(x)
    
    x = layers.Dense(128, activation='relu', name='dense2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    
    output = layers.Dense(1, activation='linear', name='aqi_output')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    return model

# ==========================================
# 3. DIAGNOSTIC ENGINE
# ==========================================
def run_audit():
    print("="*80)
    print("🕵️  INITIATING FORENSIC STRESS TEST")
    print(f"🔧  TensorFlow Version: {tf.__version__}")
    print("="*80)

    # Load Model
    try:
        model = build_model()
        # Dummy pass to initialize
        model(tf.zeros((1, 224, 224, 3)))
        model.load_weights(MODEL_WEIGHTS)
        print(f"✅ Loaded Stage 2 Weights: {os.path.basename(MODEL_WEIGHTS)}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load model.\n{e}")
        return

    print("\n⚡ RUNNING DIAGNOSTICS...\n")

    overall_report = {}

    for zone_name, config in STRESS_ZONES.items():
        dir_path = config['path']
        print(f"📂 Scanning Zone: {zone_name}")
        
        if not os.path.exists(dir_path):
            print(f"   ⚠️  PATH NOT FOUND: {dir_path}")
            continue

        # Collect images
        valid_images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not valid_images:
            print("   ⚠️  No images found.")
            continue
            
        # Limit to 20 samples per zone for speed
        sample_files = valid_images[:20]
        batch_images = []
        
        for img_file in sample_files:
            try:
                full_path = os.path.join(dir_path, img_file)
                img = keras.preprocessing.image.load_img(full_path, target_size=(224, 224))
                img_arr = keras.preprocessing.image.img_to_array(img)
                batch_images.append(img_arr)
            except:
                pass

        if not batch_images:
            continue

        # Preprocess & Predict
        batch_stack = preprocess_input(np.array(batch_images))
        preds = model.predict(batch_stack, verbose=0).flatten()

        # Statistics
        avg_aqi = np.mean(preds)
        max_aqi = np.max(preds)
        min_aqi = np.min(preds)
        negatives = np.sum(preds < 0)

        # Verdict Logic
        verdict = "✅ PASS"
        fail_reason = ""
        
        # Specific Failure Checks
        if "RAIN" in zone_name and avg_aqi > config['fail_threshold']:
            verdict = "❌ FAIL"
            fail_reason = "CONFUSING RAIN FOR SMOG"
        elif "BLUR" in zone_name and avg_aqi > config['fail_threshold']:
            verdict = "❌ FAIL"
            fail_reason = "CONFUSING BLUR FOR PARTICLES"
        elif negatives > (len(preds) * 0.5):
            verdict = "⚠️ UNSTABLE"
            fail_reason = "HIGH RATE OF NEGATIVE PREDICTIONS"

        # Print Zone Report
        print(f"   Samples: {len(preds)}")
        print(f"   Avg AQI: {avg_aqi:.2f} (Expected: {config['expected']})")
        print(f"   Range:   [{min_aqi:.2f}, {max_aqi:.2f}]")
        print(f"   Negatives: {negatives}/{len(preds)}")
        print(f"   Verdict: {verdict} {fail_reason}")
        print("-" * 40)
        
        overall_report[zone_name] = verdict

    print("\n" + "="*80)
    print("🔍 FINAL AUDIT SUMMARY")
    print("="*80)
    for zone, status in overall_report.items():
        print(f"{status}: {zone}")

if __name__ == "__main__":
    run_audit()