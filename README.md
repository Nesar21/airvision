# Vision-Based Air Quality Index Estimation: A Systematic Failure Analysis

## Executive Summary

This repository presents a systematic empirical investigation into whether images can reliably estimate Air Quality Index (AQI), and under what conditions such systems fail. Rather than proposing models claiming high accuracy, this work is designed to stress-test vision-based AQI estimation across controlled stages, progressively increasing modeling complexity and realism.

**Core Focus:** Failure characterization, error structure analysis, and claim validation.

**Primary Finding:** Vision-based AQI models can appear strong in-distribution (R²=0.931) yet collapse catastrophically out-of-distribution (R²=-4.776), producing extreme and physically implausible predictions.

---

## Research Philosophy

### Core Research Question

**Can vision-based AQI estimation be trusted beyond the dataset it is trained on?**

To answer this honestly, this work deliberately:
- Evaluates models across in-distribution and out-of-distribution (OOD) settings
- Exposes raw predictions instead of hiding behind aggregate metrics
- Examines how and when different information sources (physics vs vision) dominate predictions
- Documents failure modes that are usually ignored in literature

### What This Work Is NOT Trying to Do

This work does not claim:
- Universal AQI inference from images
- Production-ready system deployment
- Strong OOD generalization capability
- Benchmark-beating performance
- That fusion "fixes" domain shift

**Avoiding these claims is intentional.**

### What This Work Actually Achieves

#### 1. Transparent Evaluation Framework
Introduces progressive evaluation strategy:
- Physics-only baselines
- Vision-only learning
- Fusion-based decision-making

This staged approach enables causal reasoning about where performance originates and where it breaks.

#### 2. Demonstrates Silent Failure Under Domain Shift
Vision-based AQI models:
- Appear strong in-distribution (R²=0.931, MAE=19.11)
- Collapse catastrophically out-of-distribution (R²=-4.776, MAE=214.38)
- Produce extreme predictions (errors >300 AQI units)

Aggregate metrics alone are insufficient to evaluate such systems.

#### 3. Shows Fusion Does Not Solve OOD Failure
Combining physics-based signals with vision-based predictions demonstrates:
- Modest in-distribution improvement in some cases
- Active performance degradation in others
- No improvement in OOD robustness

Fusion mechanism provides interpretability, not robustness.

#### 4. Shifts Contribution From Accuracy to Accountability

Primary contribution is evaluation philosophy:
- Every claim backed by raw predictions
- Every failure shown, not hidden
- Every improvement contextualized
- Limitations stated explicitly

---

## Model Architectures

### Model 1: Airvana-LiteFusion (Statistical Fusion)

**Type:** Probabilistic Inverse-Variance Weighted Fusion

**Architecture:**

```text
Input Modalities:
├── Image Modality (Primary)
│ ├── Backbone: MobileNetV3-Small
│ ├── Input: 128×128 images, ImageNet normalization
│ ├── Output: Single AQI prediction
│ └── Uncertainty: MC-Dropout (15 passes, σ estimation)
│
├── Numeric Modality (Weather + Satellite)
│ ├── Model: LightGBM 5-fold ensemble
│ ├── Features: 13 (PM2.5, PM10, temp, humidity, pressure, visibility,
│ │ wind, wind_deg, feels_like, dew_point, sat_brightness,
│ │ sat_blur, sat_color_skew)
│ └── Uncertainty: Fixed σ_num=10.0
│
└── News Modality (Text Analysis)
├── Method: Heuristic keyword scoring
├── Pollution keywords: wildfire(+40), smog(+30), dust storm(+30)
├── Clean keywords: rain(-20), clear(-15)
└── Uncertainty: Fixed σ_news=15.0

Fusion Logic (LiteFusion):
├── Variance Weighting: w_i = 1/(σ_i² + ε)
├── Fused Mean: μ_fused = Σ(w_i × μ_i) / Σ(w_i)
└── Fused Uncertainty: σ_fused² = 1 / Σ(1/σ_i²)
```

**Performance:**

| Modality | MAE | RMSE | R² | Verdict |
|----------|-----|------|-----|---------|
| Image-Only | 12.688 | 17.253 | 0.921 | **Strongest Component** |
| Numeric-Only | 52.884 | 61.323 | 0.000 | Complete Failure |
| News-Only | 145.416 | 157.817 | -5.623 | Harmful Noise |
| **Fusion (All)** | **21.035** | **25.570** | **0.826** | **Degrades vs Image-Only** |

**Critical Findings:**
- Fusion degrades performance by 65% (MAE: 12.69 → 21.04)
- R² drops from 0.921 to 0.826
- Auxiliary modalities are noise sources, not signal enhancers
- System would perform better using image-only predictions

**OOD Test (SAPID Dataset, n=456):**
- MAE: 16.03
- RMSE: 21.33
- R²: 0.851
- Mean Residual: -1.61 (systematic underestimation)

**Limitations:**
- NOT validated on night imagery
- NOT validated on extreme AQI (>200)
- NOT validated outside South Asia

---

### Model 2: Image-Only LiteFusion (Dual-Stream Fusion)

**Type:** Multi-task RGB + Depth Fusion Network

**Architecture:**

```text
Input Streams:
├── RGB Branch
│ ├── Backbone: MobileNetV3-Large (pretrained)
│ ├── Input: 256×256 RGB images
│ └── Output: 256D embeddings
│
└── Depth Branch
├── Backbone: Custom 2-layer CNN
├── Input: 256×256 depth maps (.npy)
├── Processing: AdaptiveAvgPool → Linear(32→128)
└── Output: 128D embeddings

Fusion Strategy:
├── Concatenation: [RGB:256D + Depth:128D] = 384D
├── Reduction: Conv(384D → 256D)
└── MLP Projection: Dense layers

Multi-Task Heads:
├── AQI Head: Linear(256→1)
├── PM2.5 Head: Linear(256→1)
└── PM10 Head: Linear(256→1)

Loss Function:
└── Masked Multi-Task L1 Loss
├── w_aqi = 1.0
├── w_pm25 = 0.7
└── w_pm10 = 0.7
```

**Training Configuration:**
- Dataset: TRAQID (Day/Night split)
- Cross-Validation: 4-Fold Stratified Group K-Fold
- Epochs: 50
- Batch Size: 32
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Scheduler: Cosine Annealing
- Device: MPS (MacBook Air)

**Performance (4-Fold CV):**

| Fold | AQI MAE | AQI RMSE | AQI R² | PM2.5 MAE | PM10 MAE |
|------|---------|----------|--------|-----------|----------|
| Fold 0 | 130.33 | 158.92 | **-1.95** | 99.47 | 107.85 |
| Fold 1 | 27.50 | 56.40 | 0.628 | 40.64 | 33.73 |
| Fold 2 | 129.89 | 158.51 | **-1.94** | 99.40 | 109.50 |
| Fold 3 | 9.72 | 31.15 | 0.887 | 15.68 | 19.45 |
| **Average** | **74.36** | **101.24** | **-0.594** | **63.80** | **67.63** |

**Critical Findings:**

**Catastrophic Fold Instability:**
- 50% of folds exhibit complete failure (R² < -1.9)
- Fold 0 & 2: Predictions worse than constant mean baseline
- Fold 1 & 3: Decent to strong performance
- Average R² of -0.594 indicates system unreliability

**IEEE Submission Blockers:**
- ❌ Raw predictions NOT saved (predictions_test.csv missing)
- ❌ PM2.5/PM10 RMSE and R² NOT computed
- ❌ Scatter plots and residual analysis impossible
- ❌ No Day vs Night performance stratification
- ❌ Only placeholder visualizations available

**OOD Test Set (20 samples):**
- Day samples (n=10): AQI 70.59 ± 14.35
- Night samples (n=10): AQI 78.77 ± 4.48
- Higher pollution in night samples (OOD domain shift)

**Root Cause Analysis:**
- Training logs show negative validation R² throughout (-3.5 to -3.6)
- Model predicts worse than mean during validation
- Folds 1 & 3 recover on test set, Folds 0 & 2 do not
- Suggests data leakage or severe distribution mismatch

---

### Model 3: AirVision (Multi-Stage Progressive Training)

**Type:** Hybrid Deep Learning + Handcrafted Feature Fusion

**Architecture:**

```text
Phase 1: Data Preprocessing
├── Input: 12,240 images (IND+NEP dataset)
├── PM-Dominant Filtering (excludes gas-driven samples)
├── Soft AQI Labeling (CPCB sub-index from PM2.5)
├── Triple Confidence Scoring:
│ ├── Temporal confidence (timestamp alignment)
│ ├── Twilight confidence (solar elevation)
│ └── Soft label confidence
└── Low-information image rejection

Phase 2: Hybrid Feature Engineering (524D)
├── Handcrafted Features (12D)
│ ├── Laplacian Variance (sharpness)
│ ├── Dark Channel Prior (haze indicator)
│ ├── RMS Contrast
│ ├── Saturation Statistics (mean, std)
│ ├── Edge Density (Canny)
│ ├── Sky Fraction Heuristic
│ ├── Shannon Entropy
│ ├── Brightness Distribution Moments (skew, kurtosis)
│ └── Color Channel Ratios (R/B, G/B)
│
└── Deep Embeddings (512D)
├── Model: CLIP ViT-B/32 (pretrained)
├── L2-Normalized
└── CRITICAL BUG: Only 12/512 dimensions active

Phase 3: Stratified Splitting
├── Strategy: 5-Fold Stratified Group K-Fold CV
├── Grouping: By city (prevents leakage)
├── Stratification: By AQI bins [0-50, 50-100, ..., 300+]
├── Holdout: Delhi, Mumbai, Kanpur (2,203 samples OOD)
└── Training Pool: ~10,037 samples (IND_NEP only)

Phase 4: Fusion Architecture (Phase4Model)
├── Backbone Branch
│ ├── Model: EfficientNet-B0 (TIMM)
│ └── Output: 1024D features
│
├── Engineered Projection Branch
│ ├── Input: 524D hybrid features
│ ├── Architecture: Linear(524→64) + LayerNorm + ReLU
│ └── Output: 64D projected features
│
├── Fusion Layer
│ ├── Concatenation: [1024D + 64D] = 1088D
│ └── Dropout: p=0.3
│
└── Multi-Task Heads
├── AQI Mean: Dense(1088→128→1) + ReLU
├── AQI LogVar: Dense(1088→64→1) + ReLU (heteroscedastic)
├── Haze Logits: Dense(1088→64→3) + ReLU (auxiliary)
└── Visibility: Dense(1088→64→1) + ReLU (optional)

Phase 5: 3-Stage Progressive Training
├── Stage 1: Haze Pretraining
│ ├── Loss: CrossEntropy
│ ├── Performance: Val loss=1.477, Accuracy=36.8%
│ └── Purpose: Backbone initialization
│
├── Stage 2: Soft AQI Pretraining
│ ├── Loss: Heteroscedastic (learns variance)
│ ├── Performance: Val loss=-0.797, Normalized MAE=0.090
│ └── Purpose: Robust AQI initialization
│
└── Stage 3: IND_NEP Fine-tuning
├── Sampling: Long-tail (10× weight for AQI>250)
├── Epochs: Variable (8-25 per fold)
└── Early Stopping: Patience=6

Phase 8: MC Dropout Uncertainty Quantification
├── MC Passes: 20
├── Sigma Rejection Thresholds: [∞, 100, 80, 60, 50, 40, 30, 20]
├── Automatic Filtering: Night images, low-info images
└── Coverage vs Accuracy Trade-off Analysis
```

**Performance (5-Fold CV - In-Distribution):**

| Fold | MAE | RMSE | R² | Within-25% | Within-50% |
|------|-----|------|-----|------------|------------|
| Fold 0 | 18.65 | 27.29 | 0.929 | 72.5% | 92.5% |
| Fold 1 | 18.55 | - | 0.935 | - | - |
| Fold 2 | 19.49 | - | 0.919 | - | - |
| Fold 3 | 19.22 | - | 0.934 | - | - |
| Fold 4 | 19.65 | - | 0.929 | - | - |
| **Average** | **19.11** | **27.02** | **0.931** | - | - |

**Performance (OOD - 3-City Holdout):**

| Metric | Value | Degradation vs In-Domain |
|--------|-------|--------------------------|
| MAE | 214.38 | **11.2× worse** |
| RMSE | 249.20 | **9.2× worse** |
| R² | -4.776 | **Catastrophic** |
| Within-25% | 3.06% | 96% drop |
| Within-50% | 7.43% | 92% drop |
| Mean Bias | -213.58 | Severe underestimation |

**OOD Performance by AQI Bin:**

| AQI Range | MAE | Sample Count | Error Pattern |
|-----------|-----|--------------|---------------|
| 0-50 | 80.24 | 308 | Moderate |
| 50-100 | 105.22 | 314 | High |
| 100-150 | 154.66 | 572 | Very High |
| 150-200 | 229.17 | 525 | Severe |
| 200-300 | 303.21 | 439 | Critical |
| 300+ | 431.63 | 290 | **Catastrophic** |

**Critical Failure Modes:**

**1. CLIP Embedding Collapse**
- Only 12 out of 512 dimensions contain information (std ≈ 1.0)
- Remaining 500 dimensions collapsed (std ≈ 1e-13)
- Feature vector effectively 24D instead of 524D
- Severe bug in embedding generation process

**2. Catastrophic OOD Generalization**
- R² of -4.776 means model is 5.78× worse than predicting mean
- Systematic underestimation of 213.58 AQI units
- Error grows exponentially with true AQI magnitude
- Highest errors in hazardous range (300+ AQI)

**3. Missing Holdout Results**
- Only Mumbai results available (440 samples, MAE=242.13)
- Delhi and Kanpur results missing from analysis
- Discrepancy: 2,203 holdout samples vs 440 reported

**4. Training Instability**
- First Stage 3 attempt: R²=-12.07 (catastrophic)
- Second Stage 3 attempt: R²=0.929 (strong)
- Suggests severe initialization sensitivity

**5. Phase 6 Ablation Collapse**
- Recent ablation experiments completely broken
- R² degraded to -0.53, -2.04
- MAE jumped from 19 to 50-74
- Configuration errors when disabling Phase 2 embeddings

---

### Model 4: V1 (3-Stage Physics→Vision→Fusion)

**Type:** Sequential Physics-Vision Gated Fusion

**Architecture:**

```text
Stage 1: Physics-Based Feature Extraction (8 Features)
├── Michelson Contrast
│ └── Local intensity variation (haze reduces contrast)
│
├── FFT Slope
│ ├── Frequency domain texture loss
│ ├── Ring sampling (NOT filled disks)
│ └── Log-Log power spectrum with radial averaging
│
├── Laplacian Edge Density
│ └── Edge sharpness quantification
│
├── Color Temperature
│ └── R/B ratio for atmospheric estimation
│
├── Illuminant Vector
│ └── Maximum brightness detection
│
├── Geometric Proxy
│ └── Depth approximation using vertical gradients
│
├── Specular Reflection
│ └── Highlight detection (>0.95 threshold)
│
└── Glow Dispersion
└── Dark channel prior with 15×15 morphological erosion

Implementation Details:
├── PM2.5 >250: Linear extrapolation (0.767 AQI/μg/m³, capped at 500)
├── Parallelization: 8-core processing
├── Processing Time: 2.5-3.5 hours for 23,559 images
└── Output: model4_physics_features.csv

Stage 2: Vision Transformer Model
├── Base Architecture: EfficientNetB0 or MobileViT-XS
├── Input: 224×224 images, ImageNet normalization
├── Head Architecture:
│ ├── Linear(features → 256)
│ ├── ReLU + Dropout(0.3)
│ ├── Linear(256 → 128)
│ ├── ReLU + Dropout(0.2)
│ └── Linear(128 → 1)
└── Device: MPS (Apple Silicon) with CPU fallback

Stage 2.5: Adversarial Patching (Domain Adaptation)
├── Problem: Model predicting -800 AQI on OOD images
├── Solution: Fine-tune with pseudo-labeled negative examples
│ ├── Rain → 40 AQI (200 images)
│ ├── Motion Blur → 45 AQI (100 images)
│ ├── Fog → 180 AQI (150 images)
│ └── Glare/Overexposure → 20 AQI (100 images)
├── Training: 5 epochs, batch=32, lr=1e-4
├── Backbone: FROZEN (only trains head)
└── Total Negative Examples: 690 samples

Stage 3: Gated Physics-Vision Fusion
├── Physics Branch
│ ├── Architecture: Dense(32)→ReLU→Dense(16)→ReLU→Dense(1)
│ └── Output: phy_pred
│
├── Vision Calibration
│ └── Architecture: Dense(8)→ReLU
│
├── Gate Network
│ ├── Input: Concat([physics_features, vision_features, phy_pred])
│ ├── Architecture: Dense(16)→ReLU→Dense(1)→Sigmoid
│ └── Output: gate ∈​
│
└── Fusion Formula
├── final = gate × phy_pred + (1 - gate) × vision_input
└── Output Clipping: ReLU(max=500)

Loss Configuration:
├── Multi-Output Loss
│ ├── Final Output: weight = 1.0
│ └── Auxiliary Physics: weight = 0.3
└── Optimizer: Adam(lr=1e-3)
```

**Performance (Stage 1 - Physics Only):**

| Fold | MAE | RMSE | R² | Best Epoch | Learning Rate |
|------|-----|------|-----|------------|---------------|
| 1 | 80.52 | 108.48 | 0.2352 | 47 | 0.000125 |
| 2 | 81.87 | 109.70 | 0.2180 | 49 | 0.001 |
| 3 | 79.90 | 108.10 | 0.2247 | 46 | 0.001 |
| 4 | 81.30 | 108.61 | 0.2226 | 49 | 0.001 |
| 5 | 80.71 | 107.48 | 0.2508 | 44 | 0.0005 |
| **Average** | **80.86** | **108.48** | **0.2303** | **47** | - |

**Aggregate Physics Performance (n=19,898):**
- MAE: 81.61
- RMSE: 112.81
- R²: 0.1679
- Bias: -57.66 (systematic underprediction)

**Performance (Stage 2 - Vision Only):**

**In-Distribution Validation (n=2,448):**
- MAE: 27.59
- RMSE: 39.62
- R²: 0.8543
- Bias: +8.34 (slight overestimation)

**After Patching (10 epochs):**
- Training MAE: 34.26 → 23.02
- Validation MAE: 29.26 → 18.63
- Training R²: 0.7602 → 0.9015
- Validation R²: 0.8237 → 0.9309
- **Final Best MAE: 18.63**

**OOD Test (PM25Vision Dataset, n=2,921):**
- MAE: 113.61
- RMSE: 133.42
- R²: 0.0277 (catastrophic failure)
- Bias: -11.99

**Performance (Stage 3 - Fusion):**

**Overall Metrics (n=4,762):**
- MAE: 72.66
- RMSE: 100.14
- R²: 0.3478
- Bias: -9.45

**Gate Distribution Analysis:**
- Mean Gate Value: 0.611 (slight physics bias)
- Bimodal Distribution:
  - ~1,787 samples: gate ≈ 0.0 (pure vision)
  - ~2,740 samples: gate ≈ 1.0 (pure physics)
- No smooth blending observed (binary switching)

**Performance Comparison Across Stages:**

| Stage | MAE | RMSE | R² | Bias | Verdict |
|-------|-----|------|-----|------|---------|
| Stage 1 (Physics) | 81.61 | 112.81 | 0.168 | -57.66 | Weak, systematic underprediction |
| Stage 2 (Vision - Val) | 27.59 | 39.62 | 0.854 | +8.34 | Strong in-domain |
| Stage 2 (Vision - OOD) | 113.61 | 133.42 | 0.028 | -11.99 | Catastrophic OOD collapse |
| **Stage 3 (Fusion)** | **72.66** | **100.14** | **0.348** | **-9.45** | **Degrades vs Stage 2** |

**Critical Failure Modes:**

**1. Fusion Actively Harms Performance**
- Stage 3 MAE (72.66) > Stage 2 MAE (27.59)
- 163% performance degradation
- Fusion mechanism does not improve upon best component

**2. Gate Mechanism Broken**
- Gate values collapsed to binary (0.0 or 1.0)
- No smooth blending between modalities
- Binary switching defeats fusion purpose

**3. Physics Dominance Harmful**
- When gate = 1.0, inherits Stage 1's poor R² = 0.168
- Physics predictions have -57.66 bias
- System trusts weaker model in wrong contexts

**4. Hall of Shame (Worst Predictions):**

| True AQI | Vision Pred | Physics Pred | Gate | Final Pred | Error |
|----------|-------------|--------------|------|------------|-------|
| 31.7 | 72.6 | 331.6 | 1.0 | 331.6 | **+299.9** |
| 36.0 | 15.2 | 115.4 | 1.0 | 115.4 | **+79.4** |
| 47.0 | 103.0 | 199.0 | 1.0 | 199.0 | **+152.0** |

**Pattern:** Gate incorrectly trusts physics in clean-air scenarios, leading to massive overestimation.

**5. Best Predictions All Use Vision**
- All predictions with error <1.0 show gate ≈ 0.0
- Vision model is reliable component
- Physics model adds noise, not signal

---

## Comparative Analysis

### Performance Summary

| Model | Best MAE | Best R² | OOD MAE | OOD R² | Primary Failure Mode |
|-------|----------|---------|---------|--------|----------------------|
| Airvana-LiteFusion | 12.69 | 0.921 | 16.03 | 0.851 | Fusion degrades performance |
| Image-Only | 9.72 | 0.887 | - | - | 50% fold catastrophic failure |
| AirVision | 19.11 | 0.931 | 214.38 | -4.776 | Severe OOD collapse |
| V1 | 18.63 | 0.931 | 113.61 | 0.028 | Broken gate mechanism |

### Key Insights

**1. Vision Models Work Well In-Domain**
- Best R² range: 0.887 - 0.931
- Best MAE range: 9.72 - 19.11 AQI units
- Sufficient for controlled environments

**2. All Models Fail Out-of-Domain**
- OOD R² ranges from -4.776 to 0.851
- Catastrophic collapse in 3 out of 4 models
- Domain shift is primary challenge

**3. Fusion Does Not Solve Fundamental Problems**
- Model 1: Fusion degrades MAE from 12.69 to 21.04
- Model 4: Fusion degrades MAE from 27.59 to 72.66
- Fusion adds complexity without reliability

**4. Training Instability is Common**
- Model 2: 50% fold failure rate
- Model 3: First attempt R²=-12.07, second R²=0.929
- Model 4: Stage 2 OOD R² drops from 0.854 to 0.028

**5. Auxiliary Information is Often Harmful**
- Numeric modality (Model 1): R²=0.000
- News modality (Model 1): R²=-5.623
- Physics modality (Model 4): R²=0.168

---

## Dataset Information

### Model 1 (Airvana-LiteFusion)
- **Training Data:** Not fully specified
- **Test Data:** SAPID dataset (smartphone-based, n=456)
- **Features:** Weather (13), Satellite (3), News text

### Model 2 (Image-Only)
- **Dataset:** TRAQID (Day/Night split)
- **Training Pool:** ~70,000-100,000 images
- **Test Set:** 20 samples (10 day, 10 night)
- **Missing Images:** ~200
- **OOD Characteristics:** Night samples have higher pollution

### Model 3 (AirVision)
- **Dataset:** IND+NEP (India + Nepal)
- **Total Samples:** 12,240 (post-filtering)
- **Training Pool:** ~10,037 (IND_NEP only)
- **Holdout:** 2,203 (Delhi, Mumbai, Kanpur)
- **Gas-Driven Exclusions:** 191 samples
- **Country Groups:** IND, NEP

### Model 4 (V1)
- **Datasets:**
  - IND_NEP: 1.0 weight (primary)
  - PM25Vision_train: 0.6 weight
  - PM25Vision_test: 0.6 weight
  - TRAQID: 1.0 weight
- **Negative Examples:** 690 samples (rain, blur, fog, glare)
- **Total Processed:** 23,559 images

---

## Public Datasets and Kaggle Artifacts

All raw outputs and trained models are archived publicly on Kaggle to satisfy reproducibility requirements.

### Model 4 (V1) - Public Resources

#### Datasets

- **Stage 2 Predictions (IEEE Fix):** [https://www.kaggle.com/datasets/nesaramingad/stage2-predictions-ieee-fix](https://www.kaggle.com/datasets/nesaramingad/stage2-predictions-ieee-fix)
- **ViT Stage 2 Results:** [https://www.kaggle.com/datasets/nesaramingad/vit-stage2-result](https://www.kaggle.com/datasets/nesaramingad/vit-stage2-result)
- **ViT Stage 1 Results:** [https://www.kaggle.com/datasets/nesaramingad/vit-stage1-result](https://www.kaggle.com/datasets/nesaramingad/vit-stage1-result)
- **Stage 3 Results CSV:** [https://www.kaggle.com/datasets/nesaramingad/stage3-results-csv](https://www.kaggle.com/datasets/nesaramingad/stage3-results-csv)
- **Stage 3 Fusion Inputs:** [https://www.kaggle.com/datasets/nesaramingad/stage3-fusion-inputs](https://www.kaggle.com/datasets/nesaramingad/stage3-fusion-inputs)
- **EfficientNet-B0 ImageNet Weights:** [https://www.kaggle.com/datasets/nesaramingad/efficientnetb0-imagenet-weights-tf-2-x](https://www.kaggle.com/datasets/nesaramingad/efficientnetb0-imagenet-weights-tf-2-x)
- **Stage 2 AQI Data:** [https://www.kaggle.com/datasets/nesaramingad/stage2-aqi-data](https://www.kaggle.com/datasets/nesaramingad/stage2-aqi-data)
- **Model 4 Physics Features:** [https://www.kaggle.com/datasets/nesaramingad/model4-physics-features-v1](https://www.kaggle.com/datasets/nesaramingad/model4-physics-features-v1)

#### Training Notebooks

- **Primary Training Notebook:** [https://www.kaggle.com/code/nesaramingad/notebookae1036e3ca](https://www.kaggle.com/code/nesaramingad/notebookae1036e3ca)
- **Validation Notebook:** [https://www.kaggle.com/code/nesaramingad/notebookfba6d6a55f](https://www.kaggle.com/code/nesaramingad/notebookfba6d6a55f)
- **Testing Notebook:** [https://www.kaggle.com/code/nesaramingad/notebook9b79de3e15](https://www.kaggle.com/code/nesaramingad/notebook9b79de3e15)

#### Dataset Composition

All CSVs contain:
- `image_path`: Path to source image
- `AQI_true`: Ground truth AQI value
- `AQI_pred`: Model prediction
- `error`: Signed prediction error
- `abs_error`: Absolute prediction error
- `dataset`: Source dataset label

No post-processing was applied to predictions. All values represent raw model outputs.

---

## License

This project is licensed under the MIT License.

---

## Technical Implementation

### Training Infrastructure

**Hardware:**
- Device: MacBook Air M4 (MPS acceleration)
- CPU Fallback: Available for compatibility
- Parallelization: 8-core processing

**Software Stack:**
- Deep Learning: PyTorch, TensorFlow/Keras
- Vision: torchvision, TIMM, OpenCV, PIL
- Traditional ML: LightGBM, scikit-learn
- Data: pandas, numpy
- Embeddings: CLIP (OpenAI)

### Reproducibility

**Random Seeds:**
- Global seed: 42
- Deterministic algorithms enabled
- cuDNN benchmark disabled

**Data Splits:**
- Stratified Group K-Fold (5-fold)
- Stratification by AQI bins
- Grouping by city (prevents leakage)

**Training Configuration:**
- Batch sizes: 16-32
- Learning rates: 1e-5 to 1e-3
- Optimizers: Adam, AdamW
- Schedulers: Cosine Annealing, OneCycle, StepLR
- Early stopping: Patience 6-10 epochs

---

## Critical Limitations

### Explicit Non-Claims

This work does NOT claim:
1. Universal AQI inference from arbitrary images
2. Real-world deployment readiness
3. Strong OOD generalization capability
4. Production-level reliability
5. Robustness to domain shift
6. That fusion fixes fundamental problems

### Known Failure Modes

**1. Domain Shift Catastrophe**
- All models exhibit severe OOD performance degradation
- R² can drop from 0.931 to -4.776
- Error magnitudes increase 11× or more

**2. Training Instability**
- 50% fold failure rate in Model 2
- R² swings from -12.07 to 0.929 in Model 3
- Extreme sensitivity to initialization

**3. CLIP Embedding Collapse**
- Only 12/512 dimensions active in Model 3
- Feature extraction pipeline contains critical bugs
- Effective dimensionality 95% lower than intended

**4. Fusion Mechanisms Broken**
- Statistical fusion (Model 1) degrades by 65%
- Gated fusion (Model 4) uses binary switching instead of blending
- Fusion adds complexity without reliability gains

**5. Auxiliary Modalities Harmful**
- Numeric features: R²=0.000 (Model 1)
- News text: R²=-5.623 (Model 1)
- Physics features: R²=0.168 (Model 4)

**6. Missing Data and Incomplete Analysis**
- Model 2: Raw predictions not saved
- Model 2: PM2.5/PM10 RMSE and R² not computed
- Model 3: Delhi and Kanpur holdout results missing
- Model 3: Only 440/2,203 OOD samples analyzed

### Scope Limitations

**Spatial:**
- Validated primarily in South Asia (India, Nepal)
- Limited to urban environments
- Camera perspectives: dashcam, smartphone, static

**Temporal:**
- Daytime images only (most models)
- No night validation (Model 1)
- No seasonal variation analysis

**Environmental:**
- Not validated on extreme AQI (>200) in some models
- Limited weather condition coverage
- No validation in clean-air regions

---

## What We Explicitly Claim

This work demonstrates that:

1. **Vision-only AQI estimation is highly sensitive to domain shift**
   - Evidence: R² drops from 0.931 to -4.776 on OOD data

2. **In-distribution accuracy does not imply generalization**
   - Evidence: 11.2× MAE increase on holdout cities

3. **Fusion can improve performance only where distributions align**
   - Evidence: Fusion degrades MAE by 65-163% in mismatched scenarios

4. **Gated fusion provides interpretability, not robustness**
   - Evidence: Gate values collapse to binary, inherit weakest component

5. **Raw prediction analysis is essential for trustworthy evaluation**
   - Evidence: Aggregate metrics hide catastrophic failures

---

## Contribution to Literature

### Why This Work Matters

**1. Negative Results are Underreported**
- Literature bias toward positive results
- Failure modes usually hidden or minimized
- This work explicitly documents what doesn't work

**2. Exposes Silent Failures**
- Shows how models appear strong in aggregate
- Reveals catastrophic individual predictions
- Demonstrates domain shift sensitivity

**3. Provides Reproducible Artifact Trail**
- Complete code and configuration
- Raw predictions and analysis scripts
- Explicit documentation of bugs and fixes

**4. Prevents Overstated Claims**
- Air quality is safety-critical domain
- False confidence can harm public health decisions
- Honest evaluation prevents premature deployment

### Alignment with IEEE/arXiv Standards

This work prioritizes:
- Reproducibility over novelty
- Transparency over performance
- Evidence over claims
- Negative results over positive spin

---

## Repository Structure

```text
├── Model1_Airvana-LiteFusion/
│ ├── litefusion_core.py # Inverse-variance fusion logic
│ ├── mobilenet_predictor.py # Image modality (MC-Dropout)
│ ├── predict_numeric.py # LightGBM ensemble (5-fold)
│ ├── text_predictor.py # News keyword analyzer
│ ├── litefusion_api.py # High-level wrapper
│ ├── models/
│ │ ├── mobilenet/ # MobileNetV3 checkpoints
│ │ ├── lightgbm/ # 5-fold boosters
│ │ └── traqid_night_aqi/ # Night model (unused)
│ ├── results/
│ │ ├── overall_metrics.txt
│ │ ├── testset_image_only_metrics.txt
│ │ ├── testset_litefusion_metrics.txt
│ │ └── full_audit_report.txt # IEEE audit
│ └── requirements.txt

├── Model2_Image-Only/
│ ├── src/
│ │ ├── data/
│ │ │ ├── dataset.py # AQIMultiDataset class
│ │ │ ├── loaders.py # DataLoader builders
│ │ │ ├── sampler.py # Stratified sampler
│ │ │ └── transforms.py # Augmentation pipeline
│ │ ├── model/
│ │ │ ├── backbone_rgb.py # MobileNetV3 wrapper
│ │ │ ├── backbone_depth.py # Depth CNN
│ │ │ ├── fusion_model.py # LiteFusionModel
│ │ │ └── losses.py # Masked multi-task L1
│ │ └── preprocessing/
│ │ ├── build_master_mastercsv.py
│ │ ├── normalize_master_csv.py
│ │ ├── merge_depth_paths.py
│ │ └── generate_depth_midas.py
│ ├── configs/
│ │ ├── cfg_kfold.yaml
│ │ ├── cfg_kfold_day.yaml
│ │ └── cfg_kfold_day_full.yaml
│ ├── kfold_train.py # 4-fold CV trainer
│ ├── litefusion_model_and_train.py # Standalone trainer
│ ├── results/
│ │ ├── results_all_folds.json
│ │ ├── complete_metrics_table.csv
│ │ ├── IEEE_MODEL3_COMPLETE_AUDIT.txt
│ │ └── metrics_bar_charts_4fold_IEEE_FIGURE.jpg
│ └── splits/
│ ├── fold0/
│ ├── fold1/
│ ├── fold2/
│ └── fold3/

├── Model3_AirVision/
│ ├── src/
│ │ ├── preprocessing/
│ │ │ ├── phase1_preprocess.py # Data cleaning
│ │ │ ├── pollutant_filter.py # PM-dominance test
│ │ │ ├── temporal_alignment.py # Confidence scoring
│ │ │ └── image_quality.py # Low-info detection
│ │ ├── features/
│ │ │ ├── phase2_features.py # Handcrafted (12D)
│ │ │ ├── phase2_embeddings.py # CLIP (512D)
│ │ │ └── phase2_finalize.py # Concatenation (524D)
│ │ ├── splits/
│ │ │ └── phase3_split.py # Stratified Group K-Fold
│ │ ├── models/
│ │ │ ├── phase4_model.py # Phase4Model architecture
│ │ │ └── phase4_model_old.py # Legacy version
│ │ └── training/
│ │ ├── phase5_stage1.py # Haze pretraining
│ │ ├── phase5_stage2.py # Soft AQI pretraining
│ │ ├── phase5_stage3.py # IND_NEP fine-tuning
│ │ └── phase8_uncertainty_rejection.py # MC Dropout
│ ├── configs/
│ │ └── ablation/
│ │ ├── A1_backbone_only.yaml
│ │ ├── A2_phase2.yaml
│ │ ├── B1_haze_unc.yaml
│ │ └── B4_aqi_only.yaml
│ ├── data/
│ │ ├── metadata_image_only.csv # 12,240 rows
│ │ ├── features_image_only.csv # 12D engineered
│ │ ├── phase2_features_final_image_only.npy # 524D
│ │ ├── holdout_3city.csv # 2,203 OOD samples
│ │ └── fold{0-4}_train.csv
│ ├── results/
│ │ ├── phase8_predictions.csv
│ │ ├── ood_predictions_corrected.csv
│ │ ├── ood_metrics_summary.json
│ │ ├── coverage_vs_metrics.csv
│ │ ├── fig2_stage2_validation_scatter.jpg
│ │ └── fig3_ood_scatter.jpg
│ └── run_manifest.json # Complete execution history

├── Model4_V1/
│ ├── preprocess_model4_stage1_final.py # Physics features (8D)
│ ├── create_master_stage2_csv.py # Dataset merger
│ ├── organize_negative_examples.py # 690 OOD samples
│ ├── stage2_patch.py # Domain adaptation
│ ├── train_stage3_fusion.py # Gated fusion
│ ├── test_stage3_inference.py # Live inference
│ ├── generate_plots.py # Visualization
│ ├── results/
│ │ ├── stage1_results.csv # 5-fold physics
│ │ ├── stage2_results_summary.csv
│ │ ├── stage3_test_results.csv
│ │ ├── fig0_stage1_physics_scatter.jpg
│ │ ├── fig9_stage2_training_curves.jpg
│ │ └── fig7_fusion_gate_distribution.jpg
│ └── stage2_patched/
│ └── stage2_patched_model.weights.h5

├── docs/
│ ├── ARCHITECTURE.md
│ ├── FAILURE_ANALYSIS.md
│ └── REPRODUCIBILITY.md
└── README.md # This file
```

---

## Running the Code

### Model 1: Airvana-LiteFusion

```bash
# Single-sample inference
python run_fusion_single.py

# Batch evaluation
python litefusion_api.py --config cfg.yaml
```

### Model 2: Image-Only

```bash
# 4-fold cross-validation
python kfold_train.py --config configs/cfg_kfold.yaml

# Day-only training
python kfold_train_day.py --config configs/cfg_kfold_day_full.yaml

# Single-split training
python litefusion_model_and_train.py --config configs/cfg.yaml
```

### Model 3: AirVision

```bash
# Phase 1: Preprocessing
python src/preprocessing/phase1_preprocess.py

# Phase 2: Feature extraction
python src/features/phase2_features.py
python src/features/phase2_embeddings.py
python src/features/phase2_finalize.py

# Phase 3: Split generation
python src/splits/phase3_split.py

# Phase 5: Training (3 stages)
python src/training/phase5_stage1.py --config configs/stage1.yaml
python src/training/phase5_stage2.py --config configs/stage2.yaml
python src/training/phase5_stage3.py --config configs/stage3.yaml

# Phase 8: Uncertainty quantification
python src/training/phase8_uncertainty_rejection.py
```
### Model 4: V1

```bash
# Stage 1: Physics extraction (slow, ~3 hours)
python preprocess_model4_stage1_final.py

# Stage 2: Prepare dataset
python create_master_stage2_csv.py

# Stage 2.5: Patch for OOD
python stage2_patch.py

# Stage 3: Train fusion
python train_stage3_fusion.py

# Inference
python test_stage3_inference.py --image path/to/image.jpg
```

---

## Acknowledgments
This work prioritizes scientific integrity over performance metrics. The explicit documentation of failures is intended to prevent premature deployment of unreliable systems in safety-critical air quality monitoring applications.

Primary Contribution: Honesty, structure, and evidence—not inflated performance.

---

## MIT License

Copyright (c) 2026 Nesara Amingad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contact

**Nesara Amingad**

- Email: [nesaramingad821@gmail.com](mailto:nesaramingad821@gmail.com)
- LinkedIn: [https://www.linkedin.com/in/nesar-amingad/](https://www.linkedin.com/in/nesar-amingad/)

For questions about the research methodology, failure analysis, or dataset access, please reach out via email.

---

## Citation

If you use this work to inform your research or to avoid similar failure modes, please cite:

```bibtex
@misc{aqi_failure_analysis_2026,
  title={Vision-Based Air Quality Index Estimation: A Systematic Failure Analysis},
  author={Amingad, Nesara},
  year={2026},
  note={Research emphasizing failure characterization over performance claims},
  howpublished={\url{https://github.com/Nesar21/airvision}}
}
```

---
