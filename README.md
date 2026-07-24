# CNN Segmentation — Brain Tumor Segmentation in MRIs

Convolutional neural network (U-Net) approach to **binary whole-tumor segmentation** on
multi-modal MRI data from the **BraTS2020** dataset, evaluating the effect of preprocessing
techniques (baseline, Gaussian smoothing, and intensity normalization) on segmentation
performance.

> BENG 537 – Medical Image Processing, George Mason University
> Chase L. Roby, Jimmy T. Tran

📄 **[Read the full project report (PDF)](Final%20Project%20Report.pdf)**

## Overview

Building on the work of Pereira et al. (2016), we implement a U-Net CNN with 3×3
convolutional kernels to segment gliomas from four co-registered MRI modalities
(T1, T1Gd, T2, FLAIR) merged into a four-channel input. A simple intensity-thresholding
method serves as a baseline for comparison.

Key pipeline steps:

1. Load HDF5 (`.h5`) slices from the BraTS2020 dataset
2. Group by volume (subject) **before** splitting to prevent data leakage
3. Split into train / validation / test at the volume level
4. Extract 2D axial slices; merge four modalities into a four-channel input
5. Collapse tumor sub-region masks into a single binary whole-tumor mask
6. Apply preprocessing (baseline / Gaussian σ=1 / z-score normalization)
7. Train U-Net (BCE + Dice loss, Adam, sigmoid + 0.6 threshold)
8. Evaluate with DSC, PPV, and Sensitivity

## Results (mean ± std over test slices, n = 80)

| Preprocessing        | Dice (DSC)        | PPV               | Sensitivity       |
|----------------------|-------------------|-------------------|-------------------|
| Baseline             | 0.8330 ± 0.1549   | 0.7696 ± 0.1719   | 0.9356 ± 0.1260   |
| Normalization (z)    | 0.8400 ± 0.1596   | 0.7859 ± 0.1891   | 0.9402 ± 0.0899   |
| Gaussian (σ=1)       | 0.8238 ± 0.2180   | 0.8376 ± 0.2211   | 0.8159 ± 0.2257   |

Intensity normalization gave the highest overlap/sensitivity; Gaussian smoothing gave the
highest precision. The CNN substantially outperformed the intensity-thresholding baseline
(DSC ≈ 0.006), which failed to produce meaningful segmentation.

## Repository contents

| File | Description |
|------|-------------|
| `data-inspection.ipynb`          | Dataset loading and exploratory inspection of the BraTS2020 `.h5` slices |
| `Baseline-and-preprocessing.ipynb` | Threshold-based baseline segmentation and preprocessing experiments |
| `CNN-Segmentation.ipynb`         | U-Net training, evaluation, and visualization |
| [`Final Project Report.pdf`](Final%20Project%20Report.pdf) | Full written report |

## Data

The [BraTS2020](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) dataset
(~7.6 GB) is **not** tracked in this repository. Download it from Kaggle and place it under
`data/raw/` so the notebooks resolve the following layout:

```
data/raw/
├── BraTS20 Training Metadata.csv
└── BraTS2020_training_data/
```

## Requirements

Python with PyTorch, NumPy, h5py, scikit-image, matplotlib, and Jupyter. Run the notebooks
in the order listed above.
