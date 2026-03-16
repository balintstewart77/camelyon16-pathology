# CAMELYON16 Tumour Detection Pipeline

A deep-learning based implementation of automated tumour detection from Whole Slide Images (WSIs) using the CAMELYON16 dataset, featuring a novel 4-class classification approach

## The Problem

**Pathologist shortage is critical**: 1 pathologist per 50,000+ people in many regions. Lymph node analysis is particularly time-consuming - multiple nodes per patient, each requiring careful examination, and small metastases are easily missed.

**CAMELYON16** was an international challenge to develop algorithms matching pathologist performance. This project implements a complete pipeline and introduces a novel 4-class approach that captures tissue heterogeneity and may detect subtle tumour-associated changes that traditional binary classification misses.

## Why 4 Classes?

Most approaches use binary classification (normal vs tumour). This project separates patches into 4 classes:

| Class | Description | Source |
|-------|-------------|--------|
| 0 | Normal tissue | Normal slides |
| 1 | Normal tissue | Tumour slides (0% tumour overlap) |
| 2 | Boundary tissue | Tumour slides (1-50% tumour overlap) |
| 3 | Pure tumour | Tumour slides (>50% tumour overlap) |

**Why does this matter?**

Normal tissue in tumour slides may differ from truly normal tissue due to:
- **Field cancerisation effect**: molecular changes in tissue adjacent to tumours
- **Inflammatory response**: immune cell infiltration
- **Stromal activation**: changes in supporting tissue
- **Microenvironmental changes**: altered cell signalling
- **Missed micro-metastases**: small tumours pathologists may have overlooked

Boundary regions are critical for understanding invasion patterns and detecting micro-metastases.

## Key Results

| Experiment | Classes | Val AUC | Test AUC | Test Accuracy | Finding |
|------------|---------|---------|----------|---------------|---------|
| Control | 0 vs 3 | 0.870 | 0.838 | 78.3% | Strong discrimination between normal and pure tumour |
| Boundary detection | 0 vs 2 | 0.727 | 0.633 | 58.0% | Partial generalisation with 0.09 gap |
| **Context detection** | 0 vs 1 | 0.627 | 0.494 | 48.8% | Weak validation signal does not generalise |

The context detection experiment (Class 0 vs Class 1) tests whether normal tissue from tumour slides differs detectably from normal tissue in truly normal slides (a potential "field cancerisation" effect). While the model achieves above-random performance on the validation set (AUC 0.627), this does not generalise to the held-out test set (AUC 0.494, at chance). The 0.13 gap between validation and test performance suggests the model may be learning slide-specific artifacts (staining variation, scanner differences) rather than true biological signal. This negative result is informative: if field cancerisation effects exist in this dataset, they are subtle enough to be confounded by technical variation.

The boundary detection experiment shows partial generalisation: the model detects tumour boundary regions above chance on both validation (AUC 0.727) and test (AUC 0.633), though with reduced performance on held-out data.

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
Tumour and boundary regions are rare compared to normal tissue.

**Solution**: Adaptive dense sampling
- Normal regions: 224px stride (sparse sampling for diversity)
- Boundary regions: 56px stride (4× density to capture rare class)
- Tumour regions: 112px stride (2× density)

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      S3 Storage Layer                       │
│  Training: 160 normal WSIs, 111 tumour WSIs + annotations   │
│  Test: 80 normal WSIs, 49 tumour WSIs + annotations         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Slide Processing (Per WSI)                    │
│  1. Download to /tmp                                        │
│  2. Tissue mask (HSV threshold + morphological filtering)   │
│  3. Grid sampling (adaptive stride per class)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Macenko Stain Normalisation (torchstain)          │
│  Reference slide: tumor_050.tif                             │
│  Applied per-patch before any augmentation                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              4-Class Patch Extraction                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │ Class 0 │ │ Class 1 │ │ Class 2 │ │ Class 3 │          │
│  │ Normal  │ │ Normal  │ │Boundary │ │  Pure   │          │
│  │from Norm│ │from Tum │ │ (1-50%) │ │ Tumour  │          │
│  │stride224│ │stride224│ │stride 56│ │stride112│          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Slide-Aware Chunking                            │
│  • No slide split across chunks (prevents data leakage)     │
│  • ~1000 patches per .npz file                              │
│  • Metadata tracking: slide IDs, patch counts               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           TensorFlow Training Pipeline                      │
│  • Parallel chunk loading (tf.data.interleave)              │
│  • Balanced sampling (50:50 class ratio per batch)          │
│  • Per-patch channel normalisation (zero mean, unit std)    │
│  • Augmentation: flips, rotations, brightness jitter        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Binary Classification Experiments                 │
│  Exp 2: Normal vs Pure Tumour     (strong signal)           │
│  Exp 3: Normal vs Boundary        (moderate signal)         │
│  Exp 5: Slide Context / Field     (null result — 4 models)  │
│  Evaluation: ROC-AUC, Youden's J threshold, test holdout    │
└─────────────────────────────────────────────────────────────┘
```

## Tissue Masking

Two-step process to identify valid tissue regions:

1. **Basic mask**: Convert thumbnail to grayscale, apply brightness threshold (180), edge filtering
2. **Filtered mask**: Remove high aspect-ratio artifacts (long thin objects) and small regions

This is deliberately stringent - some tissue is occasionally lost, but artifact contamination is minimised.

## Example Patch Grids

### Normal slide (Class 0 patches)
<!-- Grid view image generated in notebook 02_patch_extraction -->

### Tumour slide (3-class sampling)
<!-- 3-class grid view image generated in notebook 02_patch_extraction -->

Green = normal (Class 1), Orange = boundary (Class 2), Red = pure tumour (Class 3)

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
| normal_from_tumor | 100,011 | 56 | 111 tumour slides |
| boundary_tumor | 91,980 | 70 | 111 tumour slides |
| pure_tumor | 86,506 | 58 | 111 tumour slides |
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
# 2: Normal vs Pure Tumour (0 vs 3)  
# 3: Slide Context (0 vs 1)
# 4: Normal vs Any Tumour (0 vs 1,2,3)

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
│   ├── __init__.py         # AWS S3 access (list/download from public bucket)
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
- **Slide-level AUC**: 0.994 (binary: tumour present or not)
- **Lesion-level FROC**: ~0.8 (localisation task)

Key differences from this work:
- Challenge winners used **transfer learning** (GoogLeNet/ImageNet) - this project trains from scratch
- Winners used **millions of patches** - this project uses ~400k due to storage constraints
- Winners used **two-stage pipelines** with random forest post-processing on heatmaps
- This project focuses on the **4-class problem** which wasn't part of the original challenge

## Limitations and Negative Results

The context detection experiment (Class 0 vs 1) yielded a key negative result: while the model learns to distinguish normal tissue from tumour slides vs normal slides during training, this does not generalise to the test set. Possible explanations:

1. **Technical confounding**: the model may detect slide-level batch effects (staining intensity, scanner artifacts) rather than biological signal
2. **Overfitting**: with subtle true effects, the model may memorise slide-specific features
3. **No detectable effect**: field cancerisation changes may not be visible at this magnification or in H&E staining

This highlights the importance of held-out test evaluation when investigating subtle biological hypotheses.

## Possible Next Steps

1. **Analyse prediction patterns**: examine high-confidence predictions spatially (are they near tumour boundaries?)
2. **Full 4-class model**: train a single model on all 4 classes
3. **Attention mechanisms**: add attention layers for better boundary detection
4. **Transfer learning**: compare performance with ImageNet-pretrained backbones
5. **Multi-scale features**: incorporate context from multiple magnification levels

## Dataset

Uses the publicly available CAMELYON16 dataset:
- **Normal slides**: 160 slides of healthy lymph node tissue
- **Tumour slides**: 111 slides with metastatic breast cancer + XML annotations
- **Test set**: 129 slides (80 normal, 49 tumour)

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
