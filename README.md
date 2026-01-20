# CAMELYON16 Tumor Detection Pipeline

A deep-learning based implementation of automated tumour detection from Whole Slide Images (WSIs) using the CAMELYON16 dataset. 

## Project Overview

This project demonstrates an end-to-end pipeline for:
1. **Patch Extraction** - Extracting 224×224 patches from gigapixel WSIs
2. **4-Class Classification** - Distinguishing between:
   - Class 0: Normal tissue from normal slides
   - Class 1: Normal tissue from tumour slides  
   - Class 2: Boundary tissue (partial tumour overlap)
   - Class 3: Pure tumour tissue
3. **Binary Model Training** - Training CNN models for various classification tasks

## Key Concepts

### Why Patch-Based Processing?
WSIs are massive (up to 100,000 × 100,000 pixels). We can't load them into memory, so we:
1. Create a low-resolution tissue mask to identify valid regions
2. Sample patches from these regions
3. Process patches in manageable chunks
4. Use a four-class patch extraction methodology that enables us to experiment with different kinds of tumour detection problems

## Example Patch Grids

### Normal slide
![Grid view of sampled normal patches from a CAMELYON16 slide](assets/images/normal_slide_grid_view.png)

### Tumour slide (3-class)
![Grid view of tumour patch sampling across three classes](assets/images/tumor_slide_3_class_grid_view.png)

### Slide-Aware Chunking
To prevent data leakage between train/validation sets, we ensure:
- No slide appears in multiple chunks
- Train/val splits happen at the chunk level
- Each chunk contains patches from complete slides only

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
    class_targets={0: 50000, 1: 25000, 2: 25000, 3: 50000},
    save_path="./data/patches"
)
```

### 2. Train a Model
```python
from src.models.training import run_binary_experiment

results = run_binary_experiment(
    dataset_path="./data/patches",
    experiment_type=2,  # Normal vs Pure Tumor
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

## Experiments

| Experiment | Classes | Purpose |
|------------|---------|---------|
| Normal vs Pure Tumor | 0 vs 3 | Baseline tumor detection |
| Normal vs Boundary | 0 vs 2 | Subtle feature detection |
| Slide Context | 0 vs 1 | Can we detect tumor-adjacent tissue? |
| Normal vs Any Tumor | 0 vs (1,2,3) | Combined tumor detection |

## Dataset

Uses the publicly available CAMELYON16 dataset:
- **Normal slides**: ~160 slides of healthy lymph node tissue
- **Tumor slides**: ~110 slides with metastatic breast cancer
- **Annotations**: XML files with tumor region polygons

Data is accessed directly from AWS S3 (no credentials needed):
```
s3://camelyon-dataset/CAMELYON16/
```

## References

- [CAMELYON16 Challenge](https://camelyon16.grand-challenge.org/)
- [OpenSlide Library](https://openslide.org/)

## License

MIT License - See LICENSE file for details.
