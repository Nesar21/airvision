import os
import shutil
import pandas as pd
import glob
import random
from pathlib import Path

"""
STAGE 2 NEGATIVE EXAMPLES ORGANIZATION SCRIPT
----------------------------------------------
Samples and organizes negative examples for training
"""

BASE_DIR = "/Users/nesar/VS/V1"
NEGATIVE_BASE = os.path.join(BASE_DIR, "negative_examples")

# Create output directory
os.makedirs(NEGATIVE_BASE, exist_ok=True)

print("=" * 80)
print("ORGANIZING NEGATIVE EXAMPLES FOR STAGE 2")
print("=" * 80)

# Sampling configuration (based on inventory results)
SAMPLING_CONFIG = {
    'motion_blur': {
        'source': f"{BASE_DIR}/blur/motion_blurred",
        'target': f"{NEGATIVE_BASE}/motion_blur",
        'num_samples': 100,
        'weather_class': 6,
        'description': 'Camera motion blur'
    },
    'overexposure': {
        'source': f"{BASE_DIR}/Exposure_Errors_Dataset/INPUT_IMAGES",
        'target': f"{NEGATIVE_BASE}/overexposure",
        'num_samples': 100,
        'weather_class': 7,
        'description': 'Overexposed/glare'
    },
    'fog': {
        'source': f"{BASE_DIR}/clear_fog/foggy",
        'target': f"{NEGATIVE_BASE}/fog",
        'num_samples': 150,
        'weather_class': 1,
        'description': 'Fog/mist'
    },
    'rain': {
        'source': f"{BASE_DIR}/Multi-class Weather Dataset/Rain",
        'target': f"{NEGATIVE_BASE}/rain",
        'num_samples': 120,
        'weather_class': 2,
        'description': 'Rain/drizzle'
    },
    'cloudy': {
        'source': f"{BASE_DIR}/Multi-class Weather Dataset/Cloudy",
        'target': f"{NEGATIVE_BASE}/cloudy",
        'num_samples': 120,
        'weather_class': 3,
        'description': 'Overcast/cloudy'
    },
    'night': {
        'source': [
            f"{BASE_DIR}/day_night_images/training/night",
            f"{BASE_DIR}/day_night_images/test/night"
        ],
        'target': f"{NEGATIVE_BASE}/night",
        'num_samples': 100,
        'weather_class': 5,
        'description': 'Night/low-light'
    }
}

def get_image_files(source_path):
    """Get all image files from source path(s)"""
    if isinstance(source_path, list):
        all_images = []
        for path in source_path:
            all_images.extend(get_image_files(path))
        return all_images
    
    if not os.path.exists(source_path):
        return []
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(source_path, ext)))
    
    return images

def sample_and_copy_images(config_key, config):
    """Sample and copy images for a scenario"""
    print(f"\n{'='*80}")
    print(f"Processing: {config_key.upper()} ({config['description']})")
    print(f"{'='*80}")
    
    # Create target directory
    os.makedirs(config['target'], exist_ok=True)
    
    # Get source images
    source_images = get_image_files(config['source'])
    print(f"Found {len(source_images)} source images")
    
    # Sample
    num_to_sample = min(config['num_samples'], len(source_images))
    if num_to_sample < config['num_samples']:
        print(f"⚠️  Warning: Only {num_to_sample} available (target: {config['num_samples']})")
    
    random.seed(42)  # Reproducible sampling
    sampled_images = random.sample(source_images, num_to_sample)
    
    # Copy images
    print(f"Copying {num_to_sample} images to {config['target']}...")
    copied_count = 0
    metadata_rows = []
    
    for i, src_path in enumerate(sampled_images):
        # Generate new filename
        ext = os.path.splitext(src_path)[1]
        new_filename = f"{config_key}_{i:04d}{ext}"
        dst_path = os.path.join(config['target'], new_filename)
        
        # Copy file
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            
            # Store metadata
            metadata_rows.append({
                'image_path': dst_path,
                'original_path': src_path,
                'scenario': config_key,
                'weather_class': config['weather_class'],
                'description': config['description'],
                'is_negative': True,
                'AQI': None  # No AQI label for negative examples
            })
            
            if (i + 1) % 50 == 0:
                print(f"   Copied {i+1}/{num_to_sample}...")
        
        except Exception as e:
            print(f"   Error copying {src_path}: {e}")
    
    print(f"✅ Successfully copied {copied_count}/{num_to_sample} images")
    return metadata_rows

# Process all scenarios
all_metadata = []

for config_key, config in SAMPLING_CONFIG.items():
    metadata_rows = sample_and_copy_images(config_key, config)
    all_metadata.extend(metadata_rows)

# Create metadata DataFrame
print(f"\n{'='*80}")
print("GENERATING METADATA CSV")
print(f"{'='*80}")

df_negative = pd.DataFrame(all_metadata)

# Summary statistics
print(f"\nTotal negative examples: {len(df_negative)}")
print(f"\nBreakdown by scenario:")
for scenario in df_negative['scenario'].unique():
    count = len(df_negative[df_negative['scenario'] == scenario])
    print(f"   {scenario:20s} : {count:3d} images")

# Save metadata CSV
output_csv = os.path.join(BASE_DIR, "negative_examples_metadata.csv")
df_negative.to_csv(output_csv, index=False)
print(f"\n✅ Metadata saved to: {output_csv}")

# Verify file integrity
print(f"\n{'='*80}")
print("VERIFYING COPIED FILES")
print(f"{'='*80}")

missing_files = 0
for idx, row in df_negative.iterrows():
    if not os.path.exists(row['image_path']):
        print(f"⚠️  Missing: {row['image_path']}")
        missing_files += 1

if missing_files == 0:
    print("✅ All files verified successfully!")
else:
    print(f"⚠️  Warning: {missing_files} files missing")

print(f"\n{'='*80}")
print("ORGANIZATION COMPLETE ✅")
print(f"{'='*80}")
print(f"\nNegative examples directory: {NEGATIVE_BASE}")
print(f"Metadata CSV: {output_csv}")
print(f"Total images organized: {len(df_negative)}")
print(f"\n🚀 Next step: Create master_stage2_data.csv (merge with model4_physics_features.csv)")
print(f"{'='*80}")
