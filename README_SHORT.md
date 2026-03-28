# Vision-Based Air Quality Index Estimation  
## A Systematic Failure Analysis

---

## One-Sentence Thesis (Read This First)

**This project demonstrates that vision-based AQI estimation can achieve high in-distribution accuracy yet fail catastrophically under domain shift, and that commonly proposed fusion strategies do not reliably resolve this failure.**

---

## What This Project Is

This repository is a **systematic empirical study** of whether images can be used to estimate Air Quality Index (AQI) reliably, and **under what conditions such systems break**.

Unlike typical ML projects that optimize for benchmark performance, this work is explicitly designed to:
- stress-test assumptions,
- expose silent failure modes,
- and evaluate whether performance claims hold outside training distributions.

---

## Core Research Question

**Can vision-based AQI estimation be trusted beyond the dataset it is trained on?**

---

## Methodology Overview (Structured, Progressive)

To answer this question, the project evaluates **four modeling paradigms**, each increasing in complexity:

1. **Physics-based estimation**
2. **Vision-only learning**
3. **Statistical and gated fusion**
4. **Multi-stage hybrid systems**

Each model is evaluated under:
- in-distribution (ID) settings
- out-of-distribution (OOD) settings
- fold-level stability analysis
- raw prediction inspection (not just aggregate metrics)

---

## Key Empirical Findings (Across All Models)

### 1. In-Distribution Performance Can Be Misleading
- Vision models achieve strong ID results (R² up to **0.931**, MAE ≈ **19 AQI**)
- These results alone suggest apparent success

### 2. Catastrophic OOD Failure Is Common
- On held-out cities and datasets:
  - MAE increases **11×**
  - R² collapses to **−4.776**
  - Errors exceed **300 AQI units**
- Models systematically underpredict hazardous pollution levels

### 3. Fusion Does Not Guarantee Robustness
- Statistical fusion degrades performance by **65%**
- Gated fusion collapses to binary switching
- Auxiliary modalities (physics, numeric, news) frequently introduce noise rather than signal

### 4. Training Instability Is a Hidden Risk
- 50% fold failure in image-only models
- Extreme sensitivity to initialization
- Identical pipelines can yield R² from **−12.07** to **0.93**

### 5. Raw Predictions Reveal Failures That Metrics Hide
- Aggregate metrics conceal extreme individual errors
- Inspecting raw outputs is essential for safety-critical systems

---

## What This Project Explicitly Does NOT Claim

This work does **not** claim:
- universal AQI inference from images
- production-readiness
- strong OOD generalization
- benchmark dominance
- that fusion “fixes” domain shift

These non-claims are intentional and documented.

---

## Primary Contribution

**The contribution of this project is not higher accuracy, but higher accountability.**

Specifically, it provides:
- a transparent evaluation framework,
- documented failure modes,
- reproducible artifacts,
- and explicit negative results in a safety-critical domain.

---

## Why This Matters

Air quality estimation influences public health decisions.

A system that:
- appears accurate in benchmarks,
- but fails silently under distribution shift,

poses real-world risk.

This project demonstrates **why honest evaluation matters more than inflated performance claims**.

---

## Artifacts and Reproducibility

- Complete code and configurations are provided
- Raw predictions are archived
- Kaggle datasets store all intermediate and final outputs
- No post-processing applied to predictions

---

## Who This Project Is For

- Researchers studying robustness, OOD generalization, and evaluation methodology
- Scholarship committees assessing research maturity and integrity
- Engineers working in safety-critical ML domains
- Reviewers seeking reproducible negative results

---

## Full Documentation

For complete architectural details, metrics, datasets, and failure analysis, see:

**`README_FULL.md`**

---

## Author

**Nesara Amingad**  
Independent Research Project  
2026

---

## Citation

```bibtex
@misc{aqi_failure_analysis_2026,
  title={Vision-Based Air Quality Index Estimation: A Systematic Failure Analysis},
  author={Amingad, Nesara},
  year={2026},
  howpublished={\url{https://github.com/Nesar21/airvision}}
}
```
