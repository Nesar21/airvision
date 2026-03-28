"""
MODEL-4 Stage 1: Physics Feature Extraction (FINAL CORRECTED VERSION)
All critical fixes applied:
- CPCB PM2.5 >250 extrapolation (not compression)
- Sobel-weighted vertical proxy (not brightness-weighted)
- FFT ring sampling (not filled disks)
- Glow normalization
- IND_NEP folder mapping (FIXED: uses 'Filename' column)
- AQI clipping to [0, 500] for all datasets
- TRAQID integer image ID handling

Run on Mac M4 Air (8-core parallelization)
Estimated time: 2.5-3.5 hours for 23,559 images
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    'IND_NEP': {
        'csv_path': '/Users/nesar/VS/V1/kaggle/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/IND_and_Nep_AQI_Dataset.csv',
        'image_base': '/Users/nesar/VS/V1/kaggle/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/IND_and_NEP/',
        'weight': 1.0,
        'has_folders': True
    },
    'PM25Vision_train': {
        'csv_path': '/Users/nesar/VS/V1/pm25vision/train/metadata_india_aqi.csv',
        'image_base': '/Users/nesar/VS/V1/pm25vision/train/images/',
        'weight': 0.6,
        'has_folders': False
    },
    'PM25Vision_test': {
        'csv_path': '/Users/nesar/VS/V1/pm25vision/test/metadata_india_aqi.csv',
        'image_base': '/Users/nesar/VS/V1/pm25vision/test/images/',
        'weight': 0.6,
        'has_folders': False
    },
    'TRAQID': {
        'csv_path': '/Users/nesar/VS/V1/TRAQID_sample/TRAQID.csv',
        'image_base': '/Users/nesar/VS/V1/TRAQID_sample/Images/',
        'weight': 1.0,
        'has_folders': False,
        'needs_conversion': True
    }
}

OUTPUT_DIR = Path('/Users/nesar/VS/V1/features/')
OUTPUT_DIR.mkdir(exist_ok=True)

# IND_NEP AQI-to-folder mapping
AQI_FOLDER_MAP = [
    (0, 50, 'a_Good'),
    (51, 100, 'b_Moderate'),
    (101, 150, 'c_Unhealthy_for_Sensitive_Groups'),
    (151, 200, 'd_Unhealthy'),
    (201, 300, 'e_Very_Unhealthy'),
    (301, 500, 'f_Severe'),
]

# ============================================================
# INDIA CPCB AQI CONVERSION (CORRECTED)
# ============================================================

def pm25_to_india_aqi(pm25):
    """
    Convert PM2.5 to India CPCB AQI standard
    
    CRITICAL FIX: PM2.5 >250 uses extrapolation (not compression)
    Reference: CPCB National Air Quality Index (2014)
    """
    if pd.isna(pm25) or pm25 < 0:
        return np.nan
    
    if pm25 <= 30:
        I_Hi, I_Lo = 50, 0
        BP_Hi, BP_Lo = 30, 0
    elif pm25 <= 60:
        I_Hi, I_Lo = 100, 51
        BP_Hi, BP_Lo = 60, 31
    elif pm25 <= 90:
        I_Hi, I_Lo = 200, 101
        BP_Hi, BP_Lo = 90, 61
    elif pm25 <= 120:
        I_Hi, I_Lo = 300, 201
        BP_Hi, BP_Lo = 120, 91
    elif pm25 <= 250:
        I_Hi, I_Lo = 400, 301
        BP_Hi, BP_Lo = 250, 121
    else:
        # CRITICAL FIX: Extrapolate beyond CPCB-defined range
        # CPCB defines PM2.5 ≥250 → AQI ≥401 (open-ended)
        # Use slope from Very Poor bin for consistency
        slope = (400 - 301) / (250 - 121)  # 0.767 AQI per μg/m³
        aqi = 401 + slope * (pm25 - 251)
        return min(round(aqi, 1), 500.0)  # Cap at scale limit
    
    # Linear interpolation
    aqi = ((I_Hi - I_Lo) / (BP_Hi - BP_Lo)) * (pm25 - BP_Lo) + I_Lo
    return round(aqi, 1)

# ============================================================
# IMAGE PATH RESOLUTION (FIXED)
# ============================================================

def find_ind_nep_image(row, base_path):
    """
    Find IND_NEP image in correct AQI category folder
    
    CRITICAL FIX: Uses 'Filename' column, strips folder prefix
    """
    # Get filename from CSV
    image_name = row.get('Filename', '')
    if not image_name:
        return None
    
    # Strip any folder prefix (just get basename)
    image_name = Path(image_name).name
    
    # Get AQI to determine folder
    aqi = row['AQI']
    
    # Find correct AQI category folder
    for lo, hi, folder in AQI_FOLDER_MAP:
        if lo <= aqi <= hi:
            candidate = Path(base_path) / folder / image_name
            if candidate.exists():
                return candidate
    
    return None

def find_traqid_image(image_name, base_path):
    """
    Find TRAQID image in nested folder structure
    
    CRITICAL FIX: Handles integer image IDs
    """
    # Convert to string if integer
    image_name = str(image_name)
    
    # TRAQID structure: Images/1/Front/, Images/1/Rear/, Images/2/Front/, Images/2/Rear/
    for seq in ['1', '2']:
        for direction in ['Front', 'Rear']:
            # Try with extensions
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                candidate = Path(base_path) / seq / direction / (image_name + ext)
                if candidate.exists():
                    return candidate
            
            # Try exact match (if filename already includes extension)
            candidate = Path(base_path) / seq / direction / image_name
            if candidate.exists():
                return candidate
    
    return None

# ============================================================
# PHYSICS FEATURE EXTRACTION FUNCTIONS
# ============================================================

def sky_mask_simple(image):
    """
    Section 2.2: Sky segmentation
    Isolates top 50% of image for sky-based features
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    
    # Sky detection: bright + low saturation OR blue hue
    mask = (
        ((hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 100)) |  # Bright, desaturated
        ((hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 130))     # Blue hue range
    )
    
    # Keep only TOP 50% (sky region)
    mask[int(h * 0.5):, :] = 0
    
    return mask.astype(np.uint8)

def extract_michelson_contrast(image):
    """
    Feature 1: Michelson Contrast (local)
    Measures local intensity variation (haze reduces this)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    window_size = 32
    contrasts = []
    
    for i in range(0, gray.shape[0] - window_size, window_size):
        for j in range(0, gray.shape[1] - window_size, window_size):
            window = gray[i:i+window_size, j:j+window_size]
            I_max = window.max()
            I_min = window.min()
            
            if I_max + I_min > 0:
                contrast = (I_max - I_min) / (I_max + I_min)
                contrasts.append(contrast)
    
    return float(np.mean(contrasts)) if contrasts else 0.0

def extract_fft_slope(image):
    """
    Feature 2: FFT Power Spectrum Slope (CORRECTED)
    Measures texture loss in frequency domain
    Haze → steeper negative slope (high-freq attenuation)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    h, w = magnitude.shape
    center = (h // 2, w // 2)
    
    # CRITICAL FIX: Sample RINGS (not filled disks)
    radii = np.arange(10, min(center) - 10, 5)
    power = []
    
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    for r in radii:
        # Ring mask (thickness = 2 pixels)
        ring_mask = (dist_from_center >= r - 1) & (dist_from_center <= r + 1)
        
        if ring_mask.sum() > 0:
            power.append(np.mean(magnitude[ring_mask]))
    
    if len(power) < 2:
        return -1.5  # Default for heavily degraded images
    
    # Log-log linear fit
    log_r = np.log(radii[:len(power)] + 1)
    log_p = np.log(np.array(power) + 1)
    
    slope = np.polyfit(log_r, log_p, 1)[0]
    return float(slope)

def extract_laplacian_edge(image):
    """
    Feature 3: Laplacian Edge Sharpness
    Variance of Laplacian → edge strength (haze blurs edges)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())

def extract_color_temperature(image, sky_mask):
    """
    Feature 4: Color Temperature (sky-masked)
    Warm (red/yellow) vs Cool (blue) sky color
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if sky_mask.sum() == 0:
        return 0.5  # Neutral
    
    sky_hue = hsv[:, :, 0][sky_mask > 0]
    
    if len(sky_hue) == 0:
        return 0.5
    
    # Temperature score: 0=cool (blue), 1=warm (red/yellow)
    warm_mask = ((sky_hue < 30) | (sky_hue > 150)).sum()
    cool_mask = ((sky_hue > 90) & (sky_hue < 130)).sum()
    
    if warm_mask + cool_mask == 0:
        return 0.5
    
    temp_score = warm_mask / (warm_mask + cool_mask)
    return float(temp_score)

def extract_illuminant_vector(image):
    """
    Feature 5: Illuminant Vector (Gray-World assumption)
    R/G ratio indicates overall scene color cast
    """
    r_mean = image[:, :, 2].mean()
    g_mean = image[:, :, 1].mean()
    b_mean = image[:, :, 0].mean()
    
    # R/G ratio (warmth indicator)
    illuminant = r_mean / (g_mean + 1e-6)
    return float(illuminant)

def extract_vertical_ordering_proxy(image):
    """
    Feature 6: Vertical Ordering Proxy (SOBEL-WEIGHTED)
    
    IMPORTANT: This is a PROXY for depth, not depth estimation.
    Uses Sobel gradient magnitude (structure) instead of brightness.
    
    Rationale: 
    - Objects higher in frame = farther (perspective geometry)
    - Edge-weighted center-of-mass reduces brightness bias
    
    Future work: Replace with MiDaS relative depth (Phase 1.5)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # CRITICAL FIX: Use Sobel gradient (structure over brightness)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    
    # Vertical weights = gradient strength per row
    vertical_weights = gradient_mag.mean(axis=1)
    
    if vertical_weights.sum() == 0:
        return 0.5
    
    # Center-of-mass in vertical direction
    y_coords = np.arange(h)
    center_of_mass = np.average(y_coords, weights=vertical_weights)
    
    # Normalize to [0, 1] (0=top, 1=bottom)
    ordering_proxy = center_of_mass / h
    
    return float(ordering_proxy)

def extract_specular_reflection(image):
    """
    Feature 7: Specular Reflection (wetness detector)
    High intensity + low gradient = specular highlights (rain/wet roads)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # High intensity pixels
    high_intensity = (gray > 200).astype(np.uint8)
    
    # Gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(gx**2 + gy**2)
    
    # Low gradient pixels
    low_gradient = (gradient < 20).astype(np.uint8)
    
    # Specular = bright + smooth
    specular_mask = high_intensity * low_gradient
    specular_ratio = specular_mask.sum() / (h * w)
    
    return float(specular_ratio)

def extract_glow_dispersion(image):
    """
    Feature 8: Glow Dispersion (night halos)
    Measures light bloom around bright sources (streetlights in haze)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect bright spots (light sources)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Blur to measure halo radius
    blurred = cv2.GaussianBlur(thresh, (21, 21), 0)
    
    if (blurred > 50).sum() == 0:
        return 0.0
    
    glow_intensity = blurred[blurred > 50].mean()
    
    # CRITICAL FIX: Normalize to [0, 1]
    glow_intensity = np.clip(glow_intensity / 255.0, 0.0, 1.0)
    
    return float(glow_intensity)

def process_single_image(args):
    """
    Process one image and extract all 8 physics features
    """
    image_path, aqi, dataset_name, weight = args
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Resize to 224x224 (standard input size)
        image = cv2.resize(image, (224, 224))
        
        # Sky segmentation
        sky_mask = sky_mask_simple(image)
        
        # Extract 8 features
        features = {
            'image_path': str(image_path),
            'dataset': dataset_name,
            'AQI': aqi,
            'loss_weight': weight,
            
            # Physics features
            'michelson_contrast': extract_michelson_contrast(image),
            'fft_slope': extract_fft_slope(image),
            'laplacian_edge': extract_laplacian_edge(image),
            'color_temperature': extract_color_temperature(image, sky_mask),
            'illuminant_vector': extract_illuminant_vector(image),
            'geometric_proxy': extract_vertical_ordering_proxy(image),
            'specular_reflection': extract_specular_reflection(image),
            'glow_dispersion': extract_glow_dispersion(image)
        }
        
        return features
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# ============================================================
# MAIN PREPROCESSING PIPELINE (FIXED)
# ============================================================

def main():
    print("=" * 70)
    print("MODEL-4 PREPROCESSING — STAGE 1: Physics Feature Extraction")
    print("FINAL CORRECTED VERSION (All critical fixes applied)")
    print("=" * 70)
    
    all_tasks = []
    stats = {name: 0 for name in DATASETS.keys()}
    
    # Process each dataset
    for dataset_name, config in DATASETS.items():
        print(f"\n📂 Loading {dataset_name}...")
        
        df = pd.read_csv(config['csv_path'])
        print(f"   Found {len(df)} rows in CSV")
        
        # CRITICAL FIX: Clip AQI to [0, 500] for all datasets
        if 'AQI' in df.columns:
            df['AQI'] = df['AQI'].clip(0, 500)
        
        # Convert TRAQID PM2.5 to India AQI if needed
        if config.get('needs_conversion'):
            print("   Converting PM2.5 to India AQI (CPCB 2014)...")
            if 'PM2.5' in df.columns:
                df['AQI'] = df['PM2.5'].apply(pm25_to_india_aqi)
            elif 'pm25' in df.columns:
                df['AQI'] = df['pm25'].apply(pm25_to_india_aqi)
            else:
                print(f"   ⚠️ WARNING: No PM2.5 column found in {dataset_name}")
                continue
            
            # Clip after conversion
            df['AQI'] = df['AQI'].clip(0, 500)
            converted_count = df['AQI'].notna().sum()
            print(f"   ✅ Converted {converted_count} samples")
        
        # Find images
        for idx, row in df.iterrows():
            image_path = None
            
            if dataset_name == 'IND_NEP':
                # CRITICAL FIX: Pass entire row to find_ind_nep_image
                image_path = find_ind_nep_image(row, config['image_base'])
                
            elif 'PM25Vision' in dataset_name:
                # PM25Vision: images in flat folder
                image_name = row['filename']
                image_path = Path(config['image_base']) / image_name
                
            elif dataset_name == 'TRAQID':
                # TRAQID: images in nested folders
                image_name = row['Image']
                image_path = find_traqid_image(image_name, config['image_base'])
            
            if image_path and image_path.exists():
                all_tasks.append((image_path, row['AQI'], dataset_name, config['weight']))
                stats[dataset_name] += 1
    
    print(f"\n{'='*70}")
    print("📊 Dataset Summary:")
    for name, count in stats.items():
        print(f"   {name}: {count} images")
    print(f"   TOTAL: {len(all_tasks)} images")
    print(f"\n⏱️  Estimated time: {len(all_tasks) * 0.4 / 3600:.1f} hours (8-core parallel)")
    print(f"{'='*70}\n")
    
    # Parallel processing
    num_workers = min(cpu_count(), 8)
    print(f"🚀 Starting processing with {num_workers} workers...\n")
    
    with Pool(num_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(process_single_image, all_tasks, chunksize=50)):
            results.append(result)
            if (i + 1) % 1000 == 0:
                print(f"   Processed {i+1}/{len(all_tasks)} images...")
    
    # Filter successful results
    results = [r for r in results if r is not None]
    
    success_rate = len(results) / len(all_tasks) * 100
    print(f"\n{'='*70}")
    print(f"✅ Successfully processed: {len(results)} / {len(all_tasks)} ({success_rate:.1f}%)")
    
    if len(results) < len(all_tasks) * 0.95:
        failed_count = len(all_tasks) - len(results)
        print(f"⚠️  WARNING: {failed_count} images failed")
        print(f"   This may be due to missing files or corrupt images")
    
    # Save to CSV
    output_file = OUTPUT_DIR / 'model4_physics_features.csv'
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    
    print(f"✅ Features saved to: {output_file}")
    
    # Feature range statistics (sanity check)
    print(f"\n{'='*70}")
    print("📊 Feature Range Statistics (Sanity Check):")
    print(f"{'Feature':<30} {'Min':>10} {'Max':>10} {'Mean':>10}")
    print("-" * 70)
    
    feature_cols = ['michelson_contrast', 'fft_slope', 'laplacian_edge', 
                    'color_temperature', 'illuminant_vector', 'geometric_proxy',
                    'specular_reflection', 'glow_dispersion']
    
    for col in feature_cols:
        min_val = df_results[col].min()
        max_val = df_results[col].max()
        mean_val = df_results[col].mean()
        print(f"{col:<30} {min_val:>10.3f} {max_val:>10.3f} {mean_val:>10.3f}")
    
    # AQI distribution
    print(f"\n{'='*70}")
    print("📊 AQI Distribution:")
    print(f"   Min: {df_results['AQI'].min():.1f}")
    print(f"   Max: {df_results['AQI'].max():.1f}")
    print(f"   Mean: {df_results['AQI'].mean():.1f}")
    print(f"   Median: {df_results['AQI'].median():.1f}")
    
    print(f"\n{'='*70}")
    print("🎯 Stage 1 COMPLETE.")
    print("\n📋 Next steps:")
    print("   1. Verify feature ranges look reasonable (no NaN, no extreme outliers)")
    print("   2. Check AQI distribution matches expected dataset composition")
    print("   3. Upload CSV to Kaggle for Stage 0 (Context Classifier)")
    print("   4. Proceed to z-score normalization (Section 4)")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
