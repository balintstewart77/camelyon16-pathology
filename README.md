# CAMELYON16 Tumor Detection Pipeline

A deep-learning based implementation of automated tumour detection from Whole Slide Images (WSIs) using the CAMELYON16 dataset, featuring a novel 4-class classification approach

## The Problem

**Pathologist shortage is critical**: 1 pathologist per 50,000+ people in many regions. Lymph node analysis is particularly time-consuming—multiple nodes per patient, each requiring careful examination, and small metastases are easily missed.

**CAMELYON16** was an international challenge to develop algorithms matching pathologist performance. This project implements a complete pipeline and introduces a novel 4-class approach that captures tissue heterogeneity and may detect subtle tumor-associated changes that traditional binary classification misses.

## Why 4 Classes?

Most approaches use binary classification (normal vs tumor). This project separates patches into 4 classes:

| Class | Description | Source |
|-------|-------------|--------|
| 0 | Normal tissue | Normal slides |
| 1 | Normal tissue | Tumor slides (0% tumor overlap) |
| 2 | Boundary tissue | Tumor slides (1-50% tumor overlap) |
| 3 | Pure tumor | Tumor slides (>50% tumor overlap) |

**Why does this matter?**

Normal tissue in tumor slides may differ from truly normal tissue due to:
- **Field cancerisation effect** — molecular changes in tissue adjacent to tumors
- **Inflammatory response** — immune cell infiltration
- **Stromal activation** — changes in supporting tissue
- **Microenvironmental changes** — altered cell signalling
- **Missed micro-metastases** — small tumors pathologists may have overlooked

Boundary regions are critical for understanding invasion patterns and detecting micro-metastases.

## Key Results

| Experiment | Classes | Test Accuracy | AUC | Finding |
|------------|---------|---------------|-----|---------|
| Control | 0 vs 3 | 83.7% | 0.901 | Strong discrimination between normal and pure tumor |
| Boundary detection | 0 vs 2 | 76.0% | 0.840 | Can detect partial tumor overlap |
| **Context detection** | 0 vs 1 | 62.7% | 0.604 | Above-chance detection of tumor-adjacent normal tissue |

The context detection result (Class 0 vs Class 1) is particularly interesting—the model can distinguish normal tissue from normal slides vs normal tissue from tumor slides at above-random chance levels. This suggests detectable differences in "normal" tissue near tumors, with potential implications for early detection or risk stratification.

## Technical Challenges & Solutions

### Challenge 1: WSIs are massive
WSIs are gigapixel images (100,000+ × 100,000+ pixels, 2-5GB each). Standard deep learning models need 224×224 inputs.

**Solution**: Patch-based processing
1. Create low-resolution tissue mask to identify valid regions
2. Sample 224×224 patches from tissue regions
3. Process patches in manageable chunks

### Challenge 2: Memory constraints
~400k patches × 224×224×3 = 150GB+ if loaded into memory.

**Solution**: Slide-aware chunked dataset generation
- Process one slide at a time (download → extract → save → delete)
- Store patches in ~1000-patch chunks as compressed `.npz` files
- Stream chunks during training via `tf.data` pipeline

### Challenge 3: Data leakage
Patches from the same slide share staining characteristics, scanner artifacts, and tissue morphology. Random train/val splits would leak information.

**Solution**: Slide-aware chunking
- All patches from a single slide stay in the same chunk
- Train/val splits happen at the chunk level
- No slide ever appears in both train and validation sets

### Challenge 4: Class imbalance in patch availability
Tumor and boundary regions are rare compared to normal tissue.

**Solution**: Adaptive dense sampling
- Normal regions: 224px stride (sparse sampling for diversity)
- Boundary regions: 56px stride (4× density to capture rare class)
- Tumor regions: 112px stride (2× density)

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      S3 Storage Layer                       │
│  Training: 160 normal WSIs, 111 tumor WSIs + annotations    │
│  Test: 80 normal WSIs, 49 tumor WSIs + annotations          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Slide Processing (Per WSI)                    │
│  1. Download to /tmp  →  2. Tissue Mask  →  3. Grid Sample  │
│                          (threshold +      (adaptive stride)│
│                           filtering)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              4-Class Patch Extraction                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │ Class 0 │ │ Class 1 │ │ Class 2 │ │ Class 3 │            │
│  │ Normal  │ │ Normal  │ │Boundary │ │  Pure   │            │
│  │from Norm│ │from Tum │ │ (1-50%) │ │ Tumor   │            │
│  │stride224│ │stride224│ │stride 56│ │stride112│            │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Slide-Aware Chunking                           │
│  • No slide split across chunks (prevents data leakage)     │
│  • ~1000 patches per .npz file                              │
│  • Metadata tracking: slide IDs, patch counts               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           TensorFlow Training Pipeline                      │
│  • Parallel chunk loading (tf.data.interleave)              │
│  • Balanced sampling (50:50 class ratio)                    │
│  • Augmentation: flips, rotations, brightness               │
│  • Optional stain normalisation                             │
└─────────────────────────────────────────────────────────────┘
```

## Tissue Masking

Two-step process to identify valid tissue regions:

1. **Basic mask**: Convert thumbnail to grayscale, apply brightness threshold (180), edge filtering
2. **Filtered mask**: Remove high aspect-ratio artifacts (long thin objects) and small regions

This is deliberately stringent—some tissue is occasionally lost, but artifact contamination is minimised.

## Example Patch Grids

### Normal slide (Class 0 patches)
![Grid view of sampled normal patches from a CAMELYON16 slide](assets/images/normal_slide_grid_view.png)

### Tumor slide (3-class sampling)
![Grid view of tumor patch sampling across three classes](assets/images/tumor_slide_3_class_grid_view.png)

Green = normal (Class 1), Orange = boundary (Class 2), Red = pure tumor (Class 3)

## Model Architecture

Lightweight custom CNN (~390K parameters) designed for this task:

```python
# 4 conv blocks with progressive filter increase
Conv2D(32, 3, stride=1)  → BatchNorm → ReLU → Dropout(0.2)
Conv2D(64, 3, stride=2)  → BatchNorm → ReLU → Dropout(0.3)
Conv2D(128, 3, stride=2) → BatchNorm → ReLU → Dropout(0.3)
Conv2D(256, 3, stride=2) → BatchNorm → ReLU → Dropout(0.4)
GlobalAveragePooling2D() → Dropout(0.5) → Dense(1, sigmoid)
```

Design choices:
- **Stride 1 in first block**: Preserves maximum spatial resolution for cellular detail
- **Gentle downsampling (2×)**: Progressive complexity without aggressive pooling
- **Global Average Pooling**: Captures contextual/global tissue state over localised features
- **Heavy dropout**: Reduces overfitting risk with limited data

## Training Dataset

Generated from the full CAMELYON16 training set:

| Class | Patches | Chunks | Source |
|-------|---------|--------|--------|
| normal_from_normal | 101,724 | 74 | 147 normal slides |
| normal_from_tumor | 100,011 | 56 | 111 tumor slides |
| boundary_tumor | 91,980 | 70 | 111 tumor slides |
| pure_tumor | 86,506 | 58 | 111 tumor slides |
| **Total** | **380,221** | **258** | |

## Installation

```bash
pip install -r requirements.txt

# OpenSlide system dependency (Ubuntu/Debian)
apt-get install -y openslide-tools
```

## Quick Start

### 1. Generate Dataset
```python
from src.dataset.generator import FourClassGenerator

generator = FourClassGenerator()
dataset_path = generator.generate(
    class_targets={0: 100000, 1: 100000, 2: 100000, 3: 100000},
    save_path="./data/patches"
)
```

### 2. Train a Model
```python
from src.models.training import run_binary_experiment

# Experiment types:
# 1: Normal vs Boundary (0 vs 2)
# 2: Normal vs Pure Tumor (0 vs 3)  
# 3: Slide Context (0 vs 1)
# 4: Normal vs Any Tumor (0 vs 1,2,3)

results = run_binary_experiment(
    dataset_path="./data/patches",
    experiment_type=2,
    epochs=20
)
```

## Project Structure

```
src/
├── data/           # Data loading and preprocessing
│   ├── s3_utils.py         # AWS S3 access
│   ├── tissue_mask.py      # Tissue detection from thumbnails
│   ├── tumor_polygons.py   # XML annotation parsing
│   └── patch_extraction.py # Patch sampling and extraction
├── dataset/        # Dataset generation
│   ├── generator.py        # 4-class chunked dataset creation
│   └── tf_pipeline.py      # TensorFlow data pipeline
└── models/         # Model training
    ├── architectures.py    # CNN architectures
    └── training.py         # Training loop and evaluation
```

## Comparison to CAMELYON16 Challenge

The winning CAMELYON16 submissions achieved:
- **Slide-level AUC**: 0.994 (binary: tumor present or not)
- **Lesion-level FROC**: ~0.8 (localisation task)

Key differences from this work:
- Challenge winners used **transfer learning** (GoogLeNet/ImageNet) — this project trains from scratch
- Winners used **millions of patches** — this project uses ~400k due to storage constraints
- Winners used **two-stage pipelines** with random forest post-processing on heatmaps
- This project focuses on the **4-class problem** which wasn't part of the original challenge

## Potential Clinical Applications

The finding that normal tissue from tumor slides is partially distinguishable from truly normal tissue suggests potential applications:

1. **Risk stratification** — flagging slides that may warrant closer inspection
2. **Early detection biomarker** — identifying field effects before visible tumor formation
3. **Quality control** — detecting potential annotation gaps or missed micro-metastases

Further validation would be needed before any clinical application.

## Possible Next Steps

1. **Analyse prediction patterns** — examine high-confidence predictions spatially (are they near tumor boundaries?)
2. **Full 4-class model** — train a single model on all 4 classes
3. **Attention mechanisms** — add attention layers for better boundary detection
4. **Transfer learning** — compare performance with ImageNet-pretrained backbones
5. **Multi-scale features** — incorporate context from multiple magnification levels

## Dataset

Uses the publicly available CAMELYON16 dataset:
- **Normal slides**: 160 slides of healthy lymph node tissue
- **Tumor slides**: 111 slides with metastatic breast cancer + XML annotations
- **Test set**: 129 slides (80 normal, 49 tumor)

Data is accessed directly from AWS S3 (no credentials needed):
```
s3://camelyon-dataset/CAMELYON16/
```

## References

- [CAMELYON16 Challenge](https://camelyon16.grand-challenge.org/)
- [OpenSlide Library](https://openslide.org/)
- [Winning Solution Paper](https://www.researchgate.net/publication/317153113)

## License

MIT License - See LICENSE file for details.
