import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os
import random

print("="*80)
print("MOBILEVIT-XS OOD TESTING ON PM25VISION (10 RANDOM SAMPLES)")
print("="*80)

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"\nDevice: {device}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DEFINE MODEL ARCHITECTURE (SAME AS TRAINING)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 1: LOADING MODEL ARCHITECTURE")
print("="*80)

class MobileViTRegressor(nn.Module):
    def __init__(self, model_name='mobilevit_xs', pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            if len(features.shape) == 4:
                self.feature_dim = features.shape[1]
            else:
                self.feature_dim = features.shape[1]
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if len(features.shape) == 4 else nn.Identity(),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output.squeeze(1)

model = MobileViTRegressor(model_name='mobilevit_xs', pretrained=False)
model = model.to(device)

# Load trained checkpoint (UPDATE THIS PATH!)
checkpoint_path = "/Users/nesar/VS/V1/vit-stage2-result/stage2_best.pth"  # ADJUST IF NEEDED

if not os.path.exists(checkpoint_path):
    print(f"\n❌ ERROR: Checkpoint not found at: {checkpoint_path}")
    print("\nPlease update checkpoint_path to the correct location of stage2_best.pth")
    exit(1)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print(f"✓ Loaded trained model from: {checkpoint_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: LOAD PM25VISION METADATA
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 2: LOADING PM25VISION TEST METADATA")
print("="*80)

metadata_path = "/Users/nesar/VS/V1/pm25vision/test/metadata_india_aqi.csv"
images_dir = "/Users/nesar/VS/V1/pm25vision/test/images"

df = pd.read_csv(metadata_path)
print(f"✓ Loaded metadata: {len(df)} samples")
print(f"✓ AQI range: {df['AQI'].min():.1f} - {df['AQI'].max():.1f}")
print(f"✓ AQI mean: {df['AQI'].mean():.1f} (±{df['AQI'].std():.1f})")

# Build full paths
df['full_path'] = df['filename'].apply(lambda x: os.path.join(images_dir, x))
df['exists'] = df['full_path'].apply(os.path.exists)

valid_count = df['exists'].sum()
print(f"✓ Valid images: {valid_count} / {len(df)}")

if valid_count == 0:
    print("\n❌ ERROR: No valid images found!")
    print(f"Expected images at: {images_dir}")
    exit(1)

df = df[df['exists']].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: SELECT 10 RANDOM SAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 3: SELECTING 10 RANDOM TEST SAMPLES")
print("="*80)

test_samples = df.sample(n=10, random_state=SEED).reset_index(drop=True)

print(f"✓ Selected {len(test_samples)} random samples")
print("\nSelected samples:")
for i, row in test_samples.iterrows():
    print(f"  {i+1}. {row['filename']:30s} | True AQI: {row['AQI']:6.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: DEFINE TRANSFORM & INFERENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 4: PREPARING INFERENCE")
print("="*80)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path, model, transform, device):
    """Predict AQI for a single image"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor)
    
    return prediction.item()

print("✓ Transform and inference function ready")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: RUN INFERENCE ON 10 SAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 5: RUNNING INFERENCE")
print("="*80)

results = []

for i, row in test_samples.iterrows():
    img_path = row['full_path']
    true_aqi = row['AQI']
    
    pred_aqi = predict_image(img_path, model, transform, device)
    error = pred_aqi - true_aqi
    abs_error = abs(error)
    
    results.append({
        'filename': row['filename'],
        'true_aqi': true_aqi,
        'pred_aqi': pred_aqi,
        'error': error,
        'abs_error': abs_error
    })
    
    print(f"  {i+1}. {row['filename']:30s} | True: {true_aqi:6.1f} | Pred: {pred_aqi:6.1f} | Error: {error:+7.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: COMPUTE METRICS & SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 6: RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)

mae = results_df['abs_error'].mean()
rmse = np.sqrt((results_df['error']**2).mean())
mean_error = results_df['error'].mean()
std_error = results_df['error'].std()

# R² calculation
y_true = results_df['true_aqi'].values
y_pred = results_df['pred_aqi'].values
ss_res = np.sum((y_true - y_pred)**2)
ss_tot = np.sum((y_true - np.mean(y_true))**2)
r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

print(f"\n📊 OOD PERFORMANCE METRICS (PM25VISION TEST):")
print(f"   MAE (Mean Absolute Error):  {mae:7.2f} AQI units")
print(f"   RMSE (Root Mean Squared):   {rmse:7.2f} AQI units")
print(f"   Mean Error (Bias):          {mean_error:+7.2f} AQI units")
print(f"   Std Error:                  {std_error:7.2f} AQI units")
print(f"   R² Score:                   {r2:7.4f}")

print(f"\n📈 ERROR DISTRIBUTION:")
print(f"   Min Error:  {results_df['error'].min():+7.1f}")
print(f"   25% Error:  {results_df['error'].quantile(0.25):+7.1f}")
print(f"   50% Error:  {results_df['error'].quantile(0.50):+7.1f}")
print(f"   75% Error:  {results_df['error'].quantile(0.75):+7.1f}")
print(f"   Max Error:  {results_df['error'].max():+7.1f}")

print(f"\n🎯 COMPARISON WITH IN-DISTRIBUTION (IND_NEP):")
print(f"   In-Dist Val MAE:  18.63 AQI units  (trained on IND_NEP)")
print(f"   OOD Test MAE:     {mae:.2f} AQI units  (tested on PM25VISION)")
print(f"   Performance Drop: {mae - 18.63:+.2f} AQI units ({(mae/18.63 - 1)*100:+.1f}%)")

print("\n" + "="*80)
print("✅ TESTING COMPLETE!")
print("="*80)
