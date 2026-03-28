"""
================================================================================
STAGE 3: LIVE INFERENCE (MATH PATCHED)
Objective: Run Fusion Model with CORRECT Feature Extraction logic.
================================================================================
"""
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = "/Users/nesar/VS/V1"
TEST_IMG_DIR = os.path.join(BASE_DIR, "testing")
MODEL_PATH = os.path.join(BASE_DIR, "stage3_results", "stage3_fusion_model.h5")
PHYSICS_CSV_PATH = os.path.join(BASE_DIR, "features", "model4_physics_features.csv")
TEST_IMAGES = ['1.jpeg', '2.jpg', '3.jpg', '5.jpeg'] 

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
def build_fusion_model(n_phy=8):
    phy_in = layers.Input(shape=(n_phy,), name="physics_input")
    vis_in = layers.Input(shape=(1,), name="vision_input")

    p = layers.Dense(32, activation="relu")(phy_in)
    p = layers.Dense(16, activation="relu")(p)
    phy_pred = layers.Dense(1, name="aux_physics_output")(p)

    v = layers.Dense(8, activation="relu")(vis_in)

    gate_x = layers.Concatenate()([p, v, phy_in])
    gate_x = layers.Dense(16, activation="relu")(gate_x)
    gate = layers.Dense(1, activation="sigmoid", name="gate_weight")(gate_x)

    fused = layers.Add()([
        layers.Multiply()([gate, phy_pred]),
        layers.Multiply()([1.0 - gate, vis_in])
    ])
    final_out = layers.ReLU(max_value=500.0, name="final_output")(fused)
    return keras.Model(inputs=[phy_in, vis_in], outputs=[final_out, phy_pred])

# ==========================================
# 3. PHYSICS FEATURE EXTRACTOR (CORRECTED)
# ==========================================
def get_fft_slope(img_gray):
    """Calculates the slope of the Log-Log power spectrum (Matches Training Data)"""
    rows, cols = img_gray.shape
    # Compute FFT
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)

    # Calculate radial average
    center_x, center_y = cols // 2, rows // 2
    y, x = np.ogrid[:rows, :cols]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.astype(int)

    # Bin the magnitudes by radius
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-6)

    # Fit line to Log-Log (exclude DC component at r=0)
    # We only care about the linear region, usually middle frequencies
    x_axis = np.arange(1, len(radial_profile))
    y_axis = radial_profile[1:]
    
    # Simple linear regression on the profile
    # Slope is usually negative (-1 to -3)
    slope, _, _, _, _ = stats.linregress(np.log(x_axis + 1e-6), y_axis)
    
    # Handle NaN
    if np.isnan(slope): slope = -1.0
    return slope

def extract_physics_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # Convert 0-1 float for colors
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    # Keep 0-255 uint8 for Grayscale/Edge ops (Standard OpenCV behavior)
    img_gray_uint8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Michelson Contrast (0.0 - 1.0)
    contrast = (np.max(img_rgb) - np.min(img_rgb)) / (np.max(img_rgb) + np.min(img_rgb) + 1e-6)
    
    # 2. FFT Slope (Now Corrected)
    fft_slope = get_fft_slope(img_gray_uint8)
    
    # 3. Laplacian Edge (Variance of Laplacian on uint8 image usually gives 0-5000 range)
    edge = cv2.Laplacian(img_gray_uint8, cv2.CV_64F).var()
    
    # 4. Color Temp
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    color_temp = np.mean(r) / (np.mean(b) + 1e-6)
    
    # 5. Illuminant (Max brightness)
    illuminant = np.max(img_rgb)
    
    # 6. Geometric Proxy (Mean depth approximation)
    h, _, _ = img_rgb.shape
    y_coords = np.linspace(0, 1, h).reshape(h, 1, 1)
    geo = np.mean(y_coords * (1 - img_rgb))
    
    # 7. Specular
    specular = np.mean(img_rgb > 0.95)
    
    # 8. Glow Dispersion
    min_channel = np.min(img_rgb, axis=2)
    # Erode (dark channel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.morphologyEx(min_channel, cv2.MORPH_ERODE, kernel)
    glow = np.mean(dark)
    
    return [contrast, fft_slope, edge, color_temp, illuminant, geo, specular, glow]

# ==========================================
# 4. MAIN INFERENCE LOOP
# ==========================================
def main():
    print(f"🏗️  Rebuilding Fusion Architecture...")
    model = build_fusion_model(n_phy=8)
    
    print(f"⚖️  Loading Weights from: {MODEL_PATH}")
    try: model.load_weights(MODEL_PATH)
    except Exception as e: 
        print(f"❌ Error loading weights: {e}")
        return

    print("📏 Fitting Scaler on training data...")
    df_train = pd.read_csv(PHYSICS_CSV_PATH)
    feature_cols = [
        'michelson_contrast', 'fft_slope', 'laplacian_edge', 
        'color_temperature', 'illuminant_vector', 'geometric_proxy', 
        'specular_reflection', 'glow_dispersion'
    ]
    df_train.columns = df_train.columns.str.strip().str.lower()
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)
    print("✅ Scaler ready.")

    print("\n" + "="*60)
    print(f"🚀 RUNNING INFERENCE")
    print("="*60)
    
    gate_model = keras.Model(inputs=model.inputs, outputs=model.get_layer('gate_weight').output)

    for img_name in TEST_IMAGES:
        path = os.path.join(TEST_IMG_DIR, img_name)
        if not os.path.exists(path):
            print(f"⚠️ Image not found: {img_name}")
            continue
            
        # A. Extract & Scale
        feats = extract_physics_features(path)
        if feats is None: continue
        feats_scaled = scaler.transform([feats])
        
        # B. Vision Input (Simulated 150)
        vision_input = np.array([[150.0]]) 
        
        # C. Predict
        preds = model.predict([feats_scaled, vision_input], verbose=0)
        final_aqi = preds[0][0][0]
        gate_val = gate_model.predict([feats_scaled, vision_input], verbose=0)[0][0]

        print(f"📸 Image: {img_name}")
        # Show FFT Slope specifically to verify fix
        print(f"   ► Physics (Contrast, Slope, Edge): [{feats[0]:.2f}, {feats[1]:.2f}, {feats[2]:.2f}]")
        print(f"   ► Gate Value: {gate_val:.4f} (0=Vision, 1=Physics)")
        print(f"   ► FINAL PREDICTED AQI: {final_aqi:.2f}")
        print("-" * 40)

if __name__ == "__main__":
    main()