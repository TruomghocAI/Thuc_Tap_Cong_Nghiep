# Class-Incremental Multi-Organ CT Segmentation on BTCV

This repository contains notebooks for preprocessing BTCV abdominal CT data and training class-incremental multi-organ segmentation models.

The project studies continual / class-incremental segmentation where organs are learned sequentially over multiple tasks. The current codebase includes preprocessing utilities and several training notebooks:

- Data exploration and preprocessing
- Fine-tuning baseline
- Joint training reference
- Learning without Forgetting (LwF)
- MiB-style baseline
- Pixel-Aware EWC baseline

> Note: The notebooks were originally written for Google Colab. To run locally, you must replace Google Drive paths with local paths.

---

## 1. Repository Structure

```text
Thuc_Tap_Cong_Nghiep/
├── processing data/
│   └── data_exploration_and_preprocessing.ipynb
│
├── source code/
│   ├── FineTuning.ipynb
│   ├── joint_training.ipynb
│   ├── LwF_baseline.ipynb
│   ├── MiB_Baseline.ipynb
│   └── Pixel-Aware EWC.ipynb
│
├── README.md
├── LICENSE
└── .gitignore
2. Hardware Recommendation

Recommended:

OS: Ubuntu / Windows + WSL2 / Windows native
Python: 3.10
GPU: NVIDIA GPU with CUDA support
VRAM: at least 8GB recommended
RAM: 16GB or higher

CPU-only execution is possible for preprocessing and debugging, but training will be very slow.

3. Environment Setup
3.1 Clone repository
git clone https://github.com/TruomghocAI/Thuc_Tap_Cong_Nghiep.git
cd Thuc_Tap_Cong_Nghiep
3.2 Create Python environment

Using venv:

python -m venv .venv

Activate environment:

Windows:

.venv\Scripts\activate

Linux / macOS:

source .venv/bin/activate
3.3 Install dependencies

For CUDA GPU, install PyTorch following the official command for your CUDA version.

Example for CUDA 12.1:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Then install the remaining packages:

pip install segmentation-models-pytorch
pip install nibabel scipy tqdm matplotlib numpy jupyter ipykernel

Optional but recommended:

pip freeze > requirements.txt

A minimal requirements.txt can be:

torch
torchvision
segmentation-models-pytorch
nibabel
scipy
tqdm
matplotlib
numpy
jupyter
ipykernel
4. Dataset Preparation

This project uses the BTCV / Multi-Atlas Labeling Beyond the Cranial Vault abdominal CT dataset.

The original preprocessing notebook expects a file named:

RawData.zip

Original Colab path:

/content/drive/MyDrive/MultiOrganSeg/RawData.zip

For local execution, use the following structure:

Thuc_Tap_Cong_Nghiep/
└── data/
    ├── RawData.zip
    ├── raw/
    └── processed/
        ├── images/
        ├── masks/
        └── metadata.json

The BTCV raw structure is expected to be similar to:

RawData/
├── Training/
│   ├── img/
│   └── label/
└── Testing/
    └── img/

Only the 30 training CT cases with labels are used for supervised segmentation experiments.

5. Local Path Configuration

The notebooks currently contain Colab-specific paths such as:

from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = "/content/drive/MyDrive/MultiOrganSeg"

For local execution, replace them with:

from pathlib import Path

PROJECT_ROOT = Path.cwd()
DRIVE_ROOT = PROJECT_ROOT

RAW_DATA_DIR = DRIVE_ROOT / "data" / "raw"
PROCESSED_DIR = DRIVE_ROOT / "data" / "processed"
IMAGES_2D_DIR = PROCESSED_DIR / "images"
MASKS_2D_DIR = PROCESSED_DIR / "masks"
METADATA_PATH = PROCESSED_DIR / "metadata.json"

CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
LOG_DIR = DRIVE_ROOT / "logs"

for path in [RAW_DATA_DIR, PROCESSED_DIR, IMAGES_2D_DIR, MASKS_2D_DIR, CHECKPOINT_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

If a notebook uses string paths, convert them back with str(...) when needed:

IMAGES_2D_DIR = str(IMAGES_2D_DIR)
MASKS_2D_DIR = str(MASKS_2D_DIR)
METADATA_PATH = str(METADATA_PATH)
6. Preprocessing

Open and run:

processing data/data_exploration_and_preprocessing.ipynb

The preprocessing notebook performs:

Install and import required libraries.
Load RawData.zip.
Extract raw NIfTI files.
Pair CT images with label masks.
Visualize CT slices and segmentation masks.
Convert 3D CT volumes into axial 2D .npy slices.
Save processed images, masks, and metadata.

Default preprocessing configuration:

CFG = {
    "hu_min": -1000,
    "hu_max": 1000,
    "target_size": (512, 512),
    "min_fg_ratio": 0.005,
}

Each valid slice is processed as:

CT slice:
    HU clipping [-1000, 1000]
    normalization to [0, 1]
    resize to 512×512
    save as float32 .npy

Mask slice:
    resize to 512×512 using nearest-neighbor interpolation
    save as uint8 .npy with labels 0–13

Expected output:

data/processed/
├── images/
│   ├── img0001_slice_XXXX.npy
│   └── ...
├── masks/
│   ├── img0001_slice_XXXX.npy
│   └── ...
└── metadata.json
7. Important Note: 2D vs 2.5D Input

The current preprocessing notebook saves single axial slices as 2D arrays:

CT:   (512, 512)
Mask: (512, 512)

However, some training notebooks expect 2.5D input with 3 adjacent CT slices:

CT:   (3, 512, 512)
Mask: (512, 512)

Therefore, before running LwF / MiB / EWC notebooks, check the expected input shape in the dataset class.

If the training notebook expects 2.5D, preprocessing should be updated to save 3-slice stacks:

[center - 1, center, center + 1]

At volume boundaries, use clamping:

prev_idx = max(center_idx - 1, 0)
next_idx = min(center_idx + 1, num_slices - 1)

Then save CT as:

ct_25d = np.stack([slice_prev, slice_center, slice_next], axis=0)

Expected shape:

(3, 512, 512)

This step is required for training notebooks that assert in_channels = 3.

8. Training Notebooks

Training notebooks are located in:

source code/

Recommended running order:

8.1 Joint Training
source code/joint_training.ipynb

Purpose:

Trains on all selected organs together.
Can be used as an upper-bound reference.
8.2 Fine-Tuning Baseline
source code/FineTuning.ipynb

Purpose:

Sequentially trains the model on new organ groups.
Does not explicitly protect old knowledge.
Serves as the lower-bound continual learning baseline.
8.3 LwF Baseline
source code/LwF_baseline.ipynb

Purpose:

Uses a frozen teacher model from the previous task.
Trains the current student using segmentation loss plus knowledge distillation loss.
Does not use replay memory.

Typical configuration:

TRAIN_CFG = {
    "batch_size": 16,
    "lr": 3e-4,
    "n_epochs": 50,
    "save_every": 10,
    "num_workers": 2,
    "loss_alpha": 0.5,
    "encoder": "resnet34",
    "pretrained": "imagenet",
    "num_classes": 14,
    "seed": 42,
}

LWF_ALPHA = 1.0
LWF_TEMPERATURE = 2.0
8.4 MiB Baseline
source code/MiB_Baseline.ipynb

Purpose:

Implements a MiB-style class-incremental segmentation baseline.
Designed to better handle background shift compared with plain LwF-style training.
8.5 Pixel-Aware EWC
source code/Pixel-Aware EWC.ipynb

Purpose:

Implements an online diagonal empirical Fisher / EWC-style regularization baseline.
Intended as a regularization-based continual learning baseline.
9. Organ Labels

The project uses 14 segmentation classes:

Label	Organ
0	Background
1	Spleen
2	Right Kidney
3	Left Kidney
4	Gallbladder
5	Esophagus
6	Liver
7	Stomach
8	Aorta
9	IVC
10	Portal Vein
11	Pancreas
12	Right Adrenal
13	Left Adrenal
10. Example Local Running Workflow
Step 1: Prepare environment
git clone https://github.com/TruomghocAI/Thuc_Tap_Cong_Nghiep.git
cd Thuc_Tap_Cong_Nghiep

python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch nibabel scipy tqdm matplotlib numpy jupyter ipykernel
Step 2: Put dataset in place
data/RawData.zip
Step 3: Run preprocessing notebook
jupyter notebook

Open:

processing data/data_exploration_and_preprocessing.ipynb

Modify path variables for local execution, then run all cells.

Step 4: Verify processed data

Expected files:

data/processed/images/*.npy
data/processed/masks/*.npy
data/processed/metadata.json
Step 5: Run training notebook

Open one of:

source code/FineTuning.ipynb
source code/LwF_baseline.ipynb
source code/MiB_Baseline.ipynb
source code/Pixel-Aware EWC.ipynb
source code/joint_training.ipynb

Before running, update:

DRIVE_ROOT
IMAGES_2D_DIR
MASKS_2D_DIR
METADATA_PATH
CHECKPOINT_DIR
LOG_DIR

to local paths.

11. Outputs

Training notebooks usually save:

checkpoints/
logs/

Typical outputs include:

model checkpoints .pth
per-task training logs
validation metrics
Dice / IoU reports
continual learning evaluation results
12. Reproducibility Notes

For reproducible experiments:

Use a fixed random seed.
Use the same train/validation/test volume split.
Avoid slice-level data leakage.
Save metadata.json and volume_split.json.
Report environment information:
Python version
PyTorch version
CUDA version
GPU name
cuDNN version
Keep validation and test sets separated.
Use the validation set only for checkpoint selection.
Use the test set only for final reporting.

Recommended deterministic settings:

import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
13. Known Issues / Things to Check Before Running
13.1 Colab-specific code

Remove or comment out:

from google.colab import drive
drive.mount('/content/drive')

when running locally.

13.2 Hard-coded Google Drive paths

Replace paths like:

/content/drive/MyDrive/...

with local paths.

13.3 2D vs 2.5D mismatch

Check whether your training notebook expects:

(512, 512)

or:

(3, 512, 512)

If the model uses in_channels=3, the preprocessing output must be 2.5D.

13.4 Dataset not included

The BTCV raw dataset is not included in this repository. Users must download it separately and place it as:

data/RawData.zip
13.5 Large checkpoints

Model checkpoints should not be committed to GitHub. Keep them in:

checkpoints/

and add them to .gitignore if needed.

14. Citation / Acknowledgement

This codebase is for research and educational purposes on class-incremental multi-organ CT segmentation using the BTCV abdominal CT dataset.
