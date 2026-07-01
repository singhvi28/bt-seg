# Brain Tumor Segmentation (BraTS 2021) 🧠

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

This repository contains a comprehensive deep learning pipeline and an interactive web application for **3D Brain Tumor Segmentation**. The models are trained and evaluated on the **BraTS 2021 dataset**, leveraging the **MONAI** (Medical Open Network for AI) framework.

## 🌟 Key Features

- **Multiple U-Net Architectures:** Implementation and comparative analysis of different state-of-the-art segmentation networks.
- **Interactive Web App:** A user-friendly **Streamlit** dashboard (`app.py`) for inference and visualization on medical scans.
- **BraTS 2021 Integration:** Specialized pre-processing, data loading, and metric evaluation tailored for the Brain Tumor Segmentation Challenge dataset.
- **Multi-Class Segmentation:** Identifies and color-codes distinct tumor sub-regions:
  - ⬛ Background (Label 0)
  - 🟥 Necrotic Core / NCR (Label 1)
  - 🟩 Peritumoral Edema / ED (Label 2)
  - 🟦 Enhancing Tumor / ET (Label 3)

## 🏗️ Model Architectures

The repository focuses on 3D volumetric segmentation using encoder-decoder networks. All models share a baseline structure that accepts 4 input modalities (FLAIR, T1, T1ce, T2) and outputs predictions for the 4 target classes.

The specific models implemented and compared include:
1. **Base U-Net**: The standard 3D Convolutional Neural Network with symmetrical downsampling and upsampling paths, utilizing skip connections to preserve spatial information.
2. **Residual U-Net (Res-UNET)**: Incorporates residual blocks (skip connections within the convolutional layers) to mitigate the vanishing gradient problem, allowing for deeper networks and better feature extraction.
3. **Dynamic U-Net (Dyn-UNet)**: A MONAI implementation inspired by the nnU-Net architecture. It dynamically adapts its topology based on the input patch size and spacing for optimal performance.
4. **Attention U-Net**: Integrates Attention Gates within the standard U-Net skip connections. These gates learn to suppress irrelevant background regions while emphasizing salient features of the tumor sub-regions.

## 📈 Evaluation Metrics

The models were evaluated based on their **Mean Dice Score** across the validation dataset. The Dice similarity coefficient measures the spatial overlap between the model's prediction and the ground truth mask. 

| Model Architecture | Mean Dice Score |
| ------------------ | --------------- |
| **Base U-Net** | 0.7613          |
| **Residual U-Net** | 0.8133          |
| **Dynamic U-Net** | 0.8204          |
| **Attention U-Net**| **0.8418** |

*Note: The **Attention U-Net** achieved the highest performance and is the default model utilized in the Streamlit web application.*

## ⚙️ Image Processing Pipeline

Medical images, particularly multi-parametric MRIs, require robust pre-processing before being fed into a deep learning model. We utilized `monai.transforms` to build a reproducible pipeline:

### 1. Data Ingestion & Formatting
- **LoadImaged**: Loads all four NIfTI modalities (`flair`, `t1`, `t1ce`, `t2`) and the ground truth label.
- **EnsureChannelFirstd**: Reorganizes arrays into channel-first formatting expected by PyTorch.
- **Orientationd (RAS)**: Normalizes the anatomical orientation to Right-Anterior-Superior (RAS) space for consistency across different scans.
- **MapLabelValued**: Standardizes the BraTS labels by mapping label `4` to `3`.
- **ConcatItemsd**: Merges the 4 separate MR modalities into a single 4-channel multi-modal tensor.

### 2. Pre-processing & Normalization
- **NormalizeIntensityd**: Applies channel-wise Z-score normalization exclusively on non-zero regions (ignoring the empty background volume).
- **SpatialPadd**: Pads smaller volumes to a guaranteed minimum region of interest (ROI) of `128 x 128 x 64`.

### 3. Training Augmentations
During the training phase, spatial augmentations are applied on the fly to prevent overfitting and improve model generalization:
- **RandSpatialCropd**: Extracts random `128 x 128 x 64` 3D patches from the padded volume.
- **RandFlipd**: Randomly flips the patches along spatial axes with a 50% probability.
- **RandRotate90d**: Performs random 90-degree rotations.

## 📁 Repository Structure

```bash
.
├── app.py                         # Streamlit application for interactive inference
├── brats-2021-base-unet.ipynb     # Notebook: Training/Evaluation of Base U-Net
├── brats-2021-attn-unet-2.ipynb   # Notebook: Training/Evaluation of Attention U-Net
├── brats-2021-Res-UNET.ipynb      # Notebook: Training/Evaluation of Residual U-Net
├── brats-2021-dyn-unet.ipynb      # Notebook: Training/Evaluation of Dynamic U-Net
├── attention_unet_brats.pth       # Pre-trained model weights for the app
└── README.md                      # Project documentation
```

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.

Clone the repository:
```bash
git clone https://github.com/singhvi28/bt-seg.git
cd bt-seg
```

### Installation
Install the required dependencies via pip:
```bash
pip install torch torchvision torchaudio
pip install monai streamlit matplotlib pillow numpy
```
*(Note: For GPU acceleration, ensure you install the PyTorch version compatible with your CUDA toolkit.)*

## 💻 Running the Streamlit App

1. Ensure the pre-trained model weights file (`attention_unet_brats.pth`) is in the root directory. If missing, run the `brats-2021-attn-unet-2.ipynb` notebook to train and save the model.
2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to the URL provided in the terminal (usually `http://localhost:8501`).
4. Upload your multi-parametric MRI scans to view the predicted tumor segmentation masks overlaid on the original image slices.
