# MedSAM2-Guided Retinal Blood Vessel Segmentation

Comparison of three U-Net architectures for retinal blood vessel segmentation, two of
which fuse features from MedSAM2 (a medical foundation model) into the U-Net backbone:

1. **Baseline U-Net** — standard U-Net, no foundation-model component.
2. **Bottleneck Fusion U-Net** — MedSAM2 features fused at the deepest encoder layer.
3. **Multi-Scale Fusion U-Net** — MedSAM2 features fused at every encoder level plus
   the bottleneck.

A single learnable scalar gate per fusion site interpolates between U-Net and MedSAM2
features, so the fusion variants can in principle recover baseline behavior by zeroing
the gate.

---

## Author

Kenny Tran — University of Central Florida

Project repo: https://github.com/ClutchKenny/MedSAM2_Fundus

---

## Running on Google Colab

The notebook is designed to run end-to-end on Colab with a T4 GPU. The full run is
roughly 100 epochs × 3 architectures and takes a few hours on a T4.

### 1. Open the notebook in Colab

Upload `MedSAM2_Fundus.ipynb` to Colab, or open it directly from GitHub via
`File → Open notebook → GitHub`.

### 2. Switch the runtime to GPU

`Runtime → Change runtime type → Hardware accelerator: GPU (T4)`

### 3. Prepare the dataset on Google Drive

The notebook mounts your Google Drive and copies the dataset into Colab's local disk.
**Before running, place your dataset at the following path:**

```
MyDrive/
└── Retinal_Vessel_Data/
    ├── DRIVE/        (optional)
    ├── STARE/        (optional)
    └── CHASE_DB1/    (default — the headline experiments use this one)
```

You only need the dataset(s) you actually want to train on. The default is
`CHASE_DB1`, set in the **Configuration** cell as `DATASET = 'CHASE_DB1'`.

### 4. Expected layout for each dataset

The notebook's loader functions look for very specific filenames inside each dataset
folder. Place files exactly as shown below.

#### CHASE_DB1 (default)

Original images and ground-truth masks live in the same folder, no subfolders:

```
CHASE_DB1/
├── Image_01L.jpg
├── Image_01L_1stHO.png
├── Image_01R.jpg
├── Image_01R_1stHO.png
├── Image_02L.jpg
├── Image_02L_1stHO.png
└── ...
```

Pattern: every `Image_XXX.jpg` must have a matching `Image_XXX_1stHO.png` mask in the
same folder. The first 20 images are used for training, the remaining 8 for test.

#### DRIVE

DRIVE follows its standard distribution layout with `training/` and `test/`
subfolders:

DRIVE/
├── training/
│   ├── images/                 (input fundus images)
│   │   ├── 21_training.tif
│   │   ├── 22_training.tif
│   │   └── ...
│   ├── 1st_manual/             
│   │   ├── 21_manual1.gif
│   │   ├── 22_manual1.gif
│   │   └── ...
│   └── mask/                  
│       ├── 21_training_mask.gif
│       └── ...
└── test/
    ├── images/                 (input fundus images)
    │   ├── 01_test.tif
    │   └── ...
    └── mask/                   
        ├── 01_test_mask.gif
        └── ...

The loader pairs each `{n}_training.tif` (or `{n}_test.tif`) with `{n}_manual1.*` by
the leading number.

#### STARE

STARE is flat, with images in the dataset root and masks in a `labels-ah/`
subfolder:

```
STARE/
├── im0001.ppm
├── im0002.ppm
├── ...
└── labels-ah/
    ├── im0001.ah.ppm           (or im0001.ah.ppm.gz — auto-decompressed)
    ├── im0002.ah.ppm
    └── ...
```

If the masks are gzipped (`.ah.ppm.gz`), the data-prep cell decompresses them
automatically on first run. The first 10 images are used for training, the rest for
test.

### 5. Run all cells

`Runtime → Run all`. The notebook will:

1. Install the MedSAM2 package from the `bowang-lab/MedSAM2` repository.
2. Download the official `MedSAM2_latest.pt` checkpoint from Hugging Face
   (~365 MB).
3. Mount your Drive and copy the chosen dataset into `/content/data/`.
4. Build all three architectures and train each one for 100 epochs.
5. Save results: `results_<DATASET>_MedSAM2.csv`,
   `training_curves_<DATASET>_MedSAM2.png`, and
   `predictions_<DATASET>_MedSAM2.png`.

### 6. Switching to a different dataset

Change one line in the **Configuration** cell:

```python
DATASET = 'CHASE_DB1'   
```

Everything else (loaders, training loop, evaluation) routes off this string.

---

## Configuration knobs

In the **Configuration** cell you can change:

| Variable | Default | Notes |
|---|---|---|
| `DATASET` | `'CHASE_DB1'` 
| `EPOCHS` | `100` | Per-architecture training epochs. |
| `BATCH_SIZE` | `4` | Reduce to 2 if you hit OOM on a smaller GPU. |
| `LR` | `1e-4` | Adam learning rate. |

---

## Outputs

After a full run, the working directory contains:

- `results_<DATASET>_MedSAM2.csv` — final test-set Dice, IoU, sensitivity, and
  specificity for each of the three models.
- `training_curves_<DATASET>_MedSAM2.png` — train/val loss and validation Dice for
  all three models on one figure.
- `predictions_<DATASET>_MedSAM2.png` — qualitative prediction grid:
  input | ground truth | Baseline | Bottleneck | Multi-Scale.

A final code cell triggers a browser download of all three.

---

## Code attribution


**External:**
- The `bowang-lab/MedSAM2` repository (Ma et al., 2024) and the `MedSAM2_latest.pt`
  pretrained checkpoint.
- The U-Net architecture (Ronneberger et al., 2015) 
- The Dice loss formulation (standard in the segmentation literature) and the
  centerline Dice (clDice) loss formulation (Shit et al., 2021).

**Original to this project:**
- `MedSAM2FeatureExtractor` — the wrapper exposing MedSAM2's image-encoder outputs as
  multi-scale features.
- `FusionBlock` — the learnable-gate fusion module.
- `BottleneckFusionUNet` and `MultiScaleFusionUNet` — both fusion-variant
  architectures.
- All dataset loaders, FOV-mask handling, CLAHE preprocessing, augmentation, training
  loop, and evaluation code.

---

## Datasets — where to download

- **CHASE_DB1** — https://www.kaggle.com/datasets/khoongweihao/chasedb1 (28 fundus images,
  two annotators).
- **DRIVE** — https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction (40 images, requires registration).
- **STARE** — https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset (20 images with vessel masks
  from two annotators).



---

## References

Almarri, B., Naveen Kumar, B., Aditya Pai, H., Bhatia Khan, S., Asiri, F., & Mahesh, T. R. (2024). Redefining retinal vessel segmentation: empowering advanced fundus image analysis with the potential of GANs. Frontiers in Medicine, 11. https://doi.org/10.3389/fmed.2024.1470941

Jin, K., Huang, X., Zhou, J., Li, Y., Yan, Y., Sun, Y., Zhang, Q., Wang, Y., & Ye, J. (2022). FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation. Scientific Data, 9(1). https://doi.org/10.1038/s41597-022-01564-3

Khoong, W. H. (2019). CHASEDB1. Kaggle.com. https://www.kaggle.com/datasets/khoongweihao/chasedb1

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment Anything. ArXiv (Cornell University). https://doi.org/10.48550/arxiv.2304.02643

Larxel. (n.d.). DRIVE Digital Retinal Images for Vessel Extraction. Www.kaggle.com. https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction

Ma, J., Yang, Z., Kim, S., Chen, B., Baharoon, M., Fallahpour, A., Asakereh, R., Lyu, H., & Wang, B. (2025). MedSAM2: Segment Anything in 3D Medical Images and Videos. ArXiv.org. https://arxiv.org/abs/2504.03600

Nacode, V. (n.d.). STARE Dataset. Www.kaggle.com. https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset

Ronneberger, O., Fischer, P., & Brox, T. (2015, May 18). U-Net: Convolutional Networks for Biomedical Image Segmentation. ArXiv.org. https://arxiv.org/abs/1505.04597

Soni, T., Gupta, S., Bharany, S., Rehman, A. U., Ghoniem, R. M., & Taye, B. M. (2025). Retinal vessel segmentation using multi scale feature attention with MobileNetV2 encoder. Scientific Reports, 15(1). https://doi.org/10.1038/s41598-025-28707-x

Suprosanna Shit, Paetzold, J. C., Anjany Sekuboyina, Ezhov, I., Unger, A., Andrey Zhylka, Pluim, W., Bauer, U., & Menze, B. H. (2021). clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. ArXiv (Cornell University). https://doi.org/10.1109/cvpr46437.2021.01629

Zhou, Y., Chia, M. A., Wagner, S. K., Ayhan, M. S., Williamson, D. J., Struyven, R. R., Liu, T., Xu, M., Lozano, M. G., Woodward-Court, P., Kihara, Y., Altmann, A., Lee, A. Y., Topol, E. J., Denniston, A. K., Alexander, D. C., & Keane, P. A. (2023). A foundation model for generalizable disease detection from retinal images. Nature, 622, 1–8. https://doi.org/10.1038/s41586-023-06555-x
