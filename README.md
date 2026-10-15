# Brain tumor segmentation in MRI with a U-Net (BraTS2020)

A U-Net, written in PyTorch, that outlines brain tumors (gliomas) in MRI scans from the BraTS2020 dataset. Alongside the model, we tested whether simple preprocessing, Gaussian smoothing or intensity normalization, changes how well it segments.

By Chase L. Roby and Jimmy T. Tran, for BENG 537 (Medical Image Processing) at George Mason University. The full write-up is in the [project report (PDF)](Final%20Project%20Report.pdf).

## What we did

The model follows Pereira et al. (2016): a U-Net with 3×3 convolutions. Each input is a 2D axial slice with the four MRI scan types stacked as channels (T1, T1 with gadolinium contrast, T2 and FLAIR). The output is a single mask marking the whole tumor. A plain intensity threshold serves as the baseline to beat.

The step that matters most happens before any training. BraTS ships as 2D slices, and neighboring slices from the same patient look almost identical. Shuffle the slices and then split them, and the same brain lands in both the training and the test set, so the model can score well just by recognizing the patient. To avoid that, slices are grouped by patient first and the split happens at the patient level.

The pipeline, in order:

1. Load the BraTS2020 `.h5` slice files.
2. Group slices by patient, then split into training, validation and test sets.
3. Stack the four scan types into one four-channel input.
4. Merge the tumor sub-regions into a single whole-tumor mask.
5. Preprocess with one of three settings: none, Gaussian smoothing (σ = 1), or z-score normalization.
6. Train the U-Net with a combined binary cross-entropy and Dice loss and the Adam optimizer, then threshold its output at 0.6.
7. Score every test slice with Dice, PPV (precision) and sensitivity.

## Results

Mean ± standard deviation over 80 test slices:

| Preprocessing | Dice | PPV | Sensitivity |
| --- | --- | --- | --- |
| None (baseline) | 0.8330 ± 0.1549 | 0.7696 ± 0.1719 | 0.9356 ± 0.1260 |
| Z-score normalization | 0.8400 ± 0.1596 | 0.7859 ± 0.1891 | 0.9402 ± 0.0899 |
| Gaussian smoothing (σ = 1) | 0.8238 ± 0.2180 | 0.8376 ± 0.2211 | 0.8159 ± 0.2257 |

Normalization gave the best overlap and sensitivity, and Gaussian smoothing the best precision, though the gaps between the three are smaller than their standard deviations. The intensity-threshold baseline reached a Dice score of about 0.006 because it marked nearly the whole brain as tumor.

Two caveats. Hardware limited the run to 5 patient volumes (3 for training, 1 for validation, 1 for testing) and 5 epochs, and the loss was still falling when training stopped. And the averages hide some complete misses on small tumors, where a slice can score 0.

## What's in the repo

Run the notebooks in this order:

| File | What it does |
| --- | --- |
| `data-inspection.ipynb` | Loads the BraTS2020 `.h5` slices and takes a first look at them |
| `Baseline-and-preprocessing.ipynb` | The threshold baseline and the preprocessing experiments |
| `CNN-Segmentation.ipynb` | U-Net training, evaluation and figures |
| `Final Project Report.pdf` | The full written report |

## Getting the data

The [BraTS2020 dataset](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) is about 7.6 GB and isn't stored here. Download it from Kaggle and put it under `data/raw/`, so the notebooks find this layout:

```
data/raw/
├── BraTS20 Training Metadata.csv
└── BraTS2020_training_data/
```

You'll need Python with PyTorch, NumPy, h5py, scikit-image, matplotlib and Jupyter.

## What would come next

- Train on more patient volumes, on a GPU. This would help more than anything else on the list.
- Add data augmentation: flips, rotations, elastic deformation and intensity jitter.
- Train longer, with early stopping and learning-rate scheduling.
- Use a loss that penalizes missed tumor more heavily (Focal-Tversky), aimed at the small-tumor failures.
- Make runs reproducible: seed every random number generator and pin dependencies.
- Report Dice per volume as well as per slice.
- Try a 2.5D or 3D U-Net, to use context from neighboring slices.
