import pandas as pd
import numpy as np
import os

"""
CREATE MASTER STAGE 2 DATA CSV
------------------------------
Merges AQI-labeled data with negative examples
Adds weather class labels for all samples
Prepares unified dataset for Stage 2 training
"""

BASE_DIR = "/Users/nesar/VS/V1"

print("=" * 80)
print("CREATING MASTER STAGE 2 DATA CSV")
print("=" * 80)

# ==========================================
# STEP 1: Load AQI-labeled data
# ==========================================
print("\n📊 STEP 1: Loading AQI-labeled data...")

df_aqi = pd.read_csv(f"{BASE_DIR}/features/model4_physics_features.csv")
print(f"   Loaded: {len(df_aqi)} AQI-labeled samples")
print(f"   Columns: {df_aqi.columns.tolist()}")

# ==========================================
# STEP 2: Load negative examples
# ==========================================
print("\n📦 STEP 2: Loading negative examples...")

df_negative = pd.read_csv(f"{BASE_DIR}/negative_examples_metadata.csv")
print(f"   Loaded: {len(df_negative)} negative examples")
print(f"   Scenarios: {df_negative['scenario'].value_counts().to_dict()}")

# ==========================================
# STEP 3: Add weather_class to AQI data
# ==========================================
print("\n🌤️  STEP 3: Assigning weather classes to AQI-labeled data...")

# For AQI-labeled images, we'll use a simple heuristic
# Most will be labeled as "pollution" (class 4) if AQI > 150
# Otherwise, "clear" (class 0)
# This is a simplified approach; Stage 3 can refine this

def assign_weather_class_aqi(row):
    """
    Simplified weather class assignment for AQI-labeled data
    In Stage 3, we can use dual-head model to refine this
    """
    aqi = row['AQI']
    
    if aqi > 200:
        return 4  # Pollution (heavy)
    elif aqi > 100:
        return 4  # Pollution (moderate)
    else:
        return 0  # Clear (or light pollution)

df_aqi['weather_class'] = df_aqi.apply(assign_weather_class_aqi, axis=1)
df_aqi['is_negative'] = False
df_aqi['scenario'] = 'aqi_labeled'
df_aqi['description'] = 'AQI-labeled image'

print(f"   Weather class distribution in AQI data:")
print(f"      Class 0 (Clear):     {len(df_aqi[df_aqi['weather_class'] == 0]):5d}")
print(f"      Class 4 (Pollution): {len(df_aqi[df_aqi['weather_class'] == 4]):5d}")

# ==========================================
# STEP 4: Align columns for merging
# ==========================================
print("\n🔗 STEP 4: Aligning columns for merge...")

# Negative examples don't have physics features yet
# We'll add them as NaN for now (can compute later if needed)
physics_features = ['michelson_contrast', 'fft_slope', 'laplacian_edge', 
                    'color_temperature', 'illuminant_vector', 'geometric_proxy',
                    'specular_reflection', 'glow_dispersion']

for feature in physics_features:
    if feature not in df_negative.columns:
        df_negative[feature] = np.nan

# Add loss_weight to negative examples (can adjust later)
df_negative['loss_weight'] = 1.0  # Equal weight for now

# Add dataset column to negative examples
df_negative['dataset'] = 'negative_examples'

# Ensure AQI column exists in negative (already None/NaN)
if 'AQI' not in df_negative.columns:
    df_negative['AQI'] = np.nan

# ==========================================
# STEP 5: Select common columns and merge
# ==========================================
print("\n🔀 STEP 5: Merging datasets...")

common_columns = [
    'image_path', 'dataset', 'AQI', 'loss_weight',
    'weather_class', 'is_negative', 'scenario', 'description',
    'michelson_contrast', 'fft_slope', 'laplacian_edge',
    'color_temperature', 'illuminant_vector', 'geometric_proxy',
    'specular_reflection', 'glow_dispersion'
]

df_aqi_subset = df_aqi[common_columns]
df_negative_subset = df_negative[common_columns]

df_master = pd.concat([df_aqi_subset, df_negative_subset], ignore_index=True)

print(f"   Total rows in master CSV: {len(df_master)}")
print(f"   AQI-labeled: {len(df_master[df_master['is_negative'] == False])}")
print(f"   Negative examples: {len(df_master[df_master['is_negative'] == True])}")

# ==========================================
# STEP 6: Add training flags
# ==========================================
print("\n🎯 STEP 6: Adding training/validation flags...")

# Exclude TRAQID from training
df_master['exclude_from_training'] = df_master['dataset'] == 'TRAQID'

# Mark for external test
df_master['external_test'] = df_master['dataset'] == 'TRAQID'

print(f"   Samples excluded from training (TRAQID): {df_master['exclude_from_training'].sum()}")
print(f"   Samples for training: {len(df_master[~df_master['exclude_from_training']])}")

# ==========================================
# STEP 7: Add AQI bins for stratification
# ==========================================
print("\n📊 STEP 7: Adding AQI bins for stratified CV...")

def assign_aqi_bin(aqi):
    """Assign AQI to standard bins for stratification"""
    if pd.isna(aqi):
        return 'negative'  # Negative examples
    elif aqi <= 50:
        return '0-50'
    elif aqi <= 100:
        return '51-100'
    elif aqi <= 150:
        return '101-150'
    elif aqi <= 200:
        return '151-200'
    elif aqi <= 300:
        return '201-300'
    else:
        return '301-500'

df_master['aqi_bin'] = df_master['AQI'].apply(assign_aqi_bin)

print(f"   AQI bin distribution:")
for bin_name in ['0-50', '51-100', '101-150', '151-200', '201-300', '301-500', 'negative']:
    count = len(df_master[df_master['aqi_bin'] == bin_name])
    print(f"      {bin_name:15s} : {count:5d} samples")

# ==========================================
# STEP 8: Save master CSV
# ==========================================
print("\n💾 STEP 8: Saving master CSV...")

output_path = f"{BASE_DIR}/master_stage2_data.csv"
df_master.to_csv(output_path, index=False)

print(f"   ✅ Saved to: {output_path}")
print(f"   Total rows: {len(df_master)}")
print(f"   Total columns: {len(df_master.columns)}")

# ==========================================
# STEP 9: Verification and summary
# ==========================================
print("\n" + "=" * 80)
print("VERIFICATION & SUMMARY")
print("=" * 80)

print(f"\n📊 Dataset Composition:")
print(f"   IND_NEP:              {len(df_master[df_master['dataset'] == 'IND_NEP']):6d}")
print(f"   PM25Vision_train:     {len(df_master[df_master['dataset'] == 'PM25Vision_train']):6d}")
print(f"   PM25Vision_test:      {len(df_master[df_master['dataset'] == 'PM25Vision_test']):6d}")
print(f"   TRAQID (excluded):    {len(df_master[df_master['dataset'] == 'TRAQID']):6d}")
print(f"   Negative examples:    {len(df_master[df_master['dataset'] == 'negative_examples']):6d}")
print(f"   {'─'*50}")
print(f"   TOTAL:                {len(df_master):6d}")

print(f"\n🎯 Training Split:")
print(f"   Trainable samples:    {len(df_master[~df_master['exclude_from_training']]):6d}")
print(f"   Excluded (TRAQID):    {len(df_master[df_master['exclude_from_training']]):6d}")

print(f"\n🌤️  Weather Class Distribution:")
for wc in sorted(df_master['weather_class'].unique()):
    count = len(df_master[df_master['weather_class'] == wc])
    wc_name = {0: 'Clear', 1: 'Fog', 2: 'Rain', 3: 'Cloudy', 
               4: 'Pollution', 5: 'Night', 6: 'Motion Blur', 7: 'Overexposure'}.get(wc, f'Class {wc}')
    print(f"   {wc} ({wc_name:15s}): {count:6d} samples")

print(f"\n✅ Master CSV created successfully!")
print(f"\n🚀 Next steps:")
print(f"   1. Review master_stage2_data.csv for correctness")
print(f"   2. Write Stage 2 training code (multi-scale EfficientNet-B0)")
print(f"   3. Implement stratified K-fold CV")
print(f"   4. Train Stage 2 model (2-3 hours)")

print("=" * 80)
