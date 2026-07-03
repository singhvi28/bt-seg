# Brain Tumor Segmentation (BraTS 2021) 🧠

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

An end-to-end deep learning pipeline — and an interactive Streamlit app — for **3D Brain Tumour Segmentation** on the **BraTS 2021** dataset, built on **MONAI** and **PyTorch**. Four volumetric encoder–decoder architectures (U-Net, ResUNet, Attention U-Net, DynUNet) are implemented, trained under an identical protocol, and compared quantitatively and qualitatively.

Manual delineation of gliomas from multi-modal MRI is slow and subject to inter-observer variability. This project automates that process end-to-end: loading raw `.nii.gz` volumes, standardising and augmenting them with MONAI transforms, training 3D segmentation networks with a combined Dice + Cross-Entropy objective, and serving the best-performing model through a browser-based inference dashboard.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [Background](#-background--related-work)
- [Dataset](#️-dataset)
- [Model Architectures & Mathematics](#️-model-architectures--mathematics)
- [Data Preprocessing Pipeline](#️-data-preprocessing-pipeline)
- [Training Configuration](#️-training-configuration)
- [Loss Function](#-loss-function)
- [Quantitative Results](#-quantitative-results)
- [Qualitative Results](#️-qualitative-results)
- [Limitations](#️-limitations)
- [Future Directions](#-future-directions)
- [Streamlit Web Application](#-streamlit-web-application-apppy)
- [Getting Started](#-getting-started)
- [References](#-references)

---

## 🌟 Key Features
- **Four 3D architectures, fair comparison:** Base U-Net, ResUNet, Attention U-Net, and DynUNet, all sharing identical preprocessing, hyperparameters, loss, and training budget.
- **Volumetric preprocessing & augmentation:** MONAI's dictionary-based transform pipeline handles orientation standardisation, per-modality normalisation, patch sampling, and 3D flips/rotations.
- **Mixed-precision training:** `torch.amp.autocast` + `GradScaler` for faster, more memory-efficient training on GPU.
- **Sliding-window inference:** Full-resolution volumes are segmented patch-by-patch via `sliding_window_inference`, avoiding OOM errors on large brain volumes.
- **Interactive inference app:** A Streamlit dashboard (`app.py`) that loads the best model (Attention U-Net) and overlays colour-coded predictions on the original MRI slices.
- **Multi-Class Segmentation:** Identifies distinct tumour sub-regions:
  - ⬛ Background (Label 0)
  - 🟥 Necrotic Core / NCR (Label 1)
  - 🟩 Peritumoral Edema / ED (Label 2)
  - 🟦 Enhancing Tumour / ET (Label 3, remapped from raw Label 4)

---

## 📚 Background & Related Work

Brain tumour segmentation has evolved through several distinct phases, each addressed by this project's architecture choices:

| Era | Contribution | Key Idea |
| :--- | :--- | :--- |
| 2015 | **U-Net** (Ronneberger et al.) | Symmetric encoder–decoder with skip connections for precise localisation |
| 2016 | **3D U-Net / V-Net** (Milletari et al.) | Volumetric convolutions + Dice Loss for class-imbalanced segmentation |
| 2016 | **ResNet / ResUNet** (He et al.) | Residual shortcuts for stable, deeper network training |
| 2018 | **Attention U-Net** (Oktay et al.) | Attention Gates that suppress irrelevant background in skip connections |
| 2019 | **UNet++** (Zhou et al.) | Nested, dense skip pathways to close the encoder–decoder semantic gap |
| 2021 | **nnU-Net** (Isensee et al.) | Self-configuring preprocessing/training pipeline rather than novel architecture |
| 2021–22 | **SwinUNETR** (Hatamizadeh et al.) | Swin Transformer encoder for long-range 3D self-attention |

For reference, published whole-tumour (WT) Dice scores on BraTS benchmarks range from **~0.84–0.88** for early 3D U-Nets up to **0.926** for transformer-based SwinUNETR — the current frontier for BraTS 2021. This project's models, trained under a modest compute budget (15 epochs, reduced channel widths), sit below these headline numbers but reproduce the same *relative ordering*: attention and residual mechanisms outperform the vanilla baseline.

---

## 🗂️ Dataset

### Overview
The [**BraTS 2021**](https://arxiv.org/abs/2107.02314) dataset provides **1,251** multi-institutional glioma MRI cases. All volumes are preprocessed, co-registered to a standard anatomical atlas, interpolated to 1mm³ isotropic resolution, and skull-stripped.

### Multi-Modal MRI Sequences
| Modality | Abbreviation | Physical Contrast | Highlights |
| :--- | :--- | :--- | :--- |
| T1-weighted | T1 | Anatomical structure; fat bright, fluid dark | Baseline healthy tissue |
| T1 Contrast-Enhanced | T1ce | Gadolinium uptake where BBB is disrupted | Enhancing Tumour (ET) |
| T2-weighted | T2 | Fluid/oedema bright | Peritumoral oedema & broad tumour extent |
| FLAIR | FLAIR | T2 with CSF signal nulled | Non-enhancing tumour & infiltrating oedema |

### Ground Truth Labels
| Raw Label | Region | Remapped To |
| :--- | :--- | :--- |
| 0 | Background | 0 |
| 1 | Necrotic / Non-enhancing Tumour Core (NCR/NET) | 1 |
| 2 | Peritumoral Oedema (ED) | 2 |
| 4 | GD-Enhancing Tumour (ET) | 3 |

Label 4 is remapped to 3 to produce a contiguous class index `[0, 1, 2, 3]`. Three clinically meaningful evaluation regions are derived from these labels: **Whole Tumour** (WT = all non-zero labels), **Tumour Core** (TC = labels 1 + 4), and **Enhancing Tumour** (ET = label 4 only).

### Dataset Partitioning
A reproducible **positional split** (no stratification by tumour grade) is applied to the sorted patient list:

| Subset | Cases | Proportion | Purpose |
| :--- | :--- | :--- | :--- |
| Training | 1,000 | ~80% | Weight optimisation |
| Validation | 125 | ~10% | Hyperparameter tuning & early stopping |
| Testing | 126 | ~10% | Final unbiased evaluation |

---

## 🏗️ Model Architectures & Mathematics

All models take a 4-channel 3D input tensor `(B, 4, H, W, D)` — one channel per MRI modality — and output a 4-class per-voxel probability tensor `(B, 4, H, W, D)`. Final masks are produced by `argmax` over the channel dimension.

### 1. Base U-Net
The foundational model, with symmetrical downsampling/upsampling paths and skip connections that preserve fine-grained spatial information from encoder to decoder.

### 2. Residual U-Net (ResUNet)
Integrates residual units into the encoder and decoder to improve gradient flow and enable stable training of deeper networks. It learns the residual mapping instead of a direct mapping:

$$F(x) = H(x) - x \implies H(x) = F(x) + x$$

The identity shortcut preserves low-level spatial features alongside high-level semantic representations at every stage.

### 3. Attention U-Net
Introduces **Attention Gates (AGs)** into the skip connections. Rather than concatenating encoder and decoder features directly, AGs use a coarser gating signal from the decoder to selectively re-weight encoder features, computing attention coefficients $\alpha \in [0,1]$. Irrelevant background is suppressed ($\alpha \to 0$) while tumour-relevant regions are amplified ($\alpha \to 1$) — particularly valuable given how little of a brain volume actually contains tumour tissue.

### 4. Dynamic U-Net (DynUNet)
An nnU-Net-inspired architecture that adapts kernel sizes and strides to the input patch dimensions, using residual convolutional blocks with instance normalisation and LeakyReLU. It supports deep supervision during training (auxiliary losses at intermediate decoder resolutions) to enforce multi-scale feature learning, well suited to BraTS's dramatic variation in tumour sub-region scale — from the large whole-tumour extent down to the thin enhancing rim.

### Shared Architectural Hyperparameters

| Parameter | Value | Meaning |
| :--- | :--- | :--- |
| `spatial_dims` | 3 | 3D convolutions for volumetric MRI |
| `in_channels` | 4 | Four MRI modalities (FLAIR, T1, T1ce, T2) |
| `out_channels` | 4 | Four segmentation classes (background + 3 tumour regions) |
| `channels` | `(16,32,64,128,256)` | Feature map width at each encoder depth level |
| `strides` | `(2,2,2,2)` | Halves spatial resolution at each encoder stage |
| `kernel_size` | 3 | 3×3×3 local neighbourhood per convolution |
| `num_res_units`* | 2 | Residual sub-blocks per stage |
| `norm_name`** | `instance` | Instance normalisation — stable for small 3D batches |
| `deep_supervision`** | `False` | Single final output; simpler training and evaluation |

*\* Unique to ResUNet | \*\* Unique to DynUNet*

> Note: the smaller channel widths used here (max 256) are below state-of-the-art BraTS configurations (which often use up to 512), a deliberate trade-off for GPU memory constraints — see [Limitations](#️-limitations).

---

## ⚙️ Data Preprocessing Pipeline

Medical volumes are standardised using MONAI's dictionary-based transform pipeline, keeping the data flow memory-efficient by applying transforms lazily per sample.

| Transform | Key Parameters | Purpose |
| :--- | :--- | :--- |
| `LoadImaged` | `keys=[flair, t1, t1ce, t2, label]` | Reads `.nii.gz` volumes from disk |
| `EnsureChannelFirstd` | All keys | Standardises tensor layout for PyTorch |
| `Orientationd` | `axcodes="RAS"` | Aligns all volumes to the same anatomical axes |
| `MapLabelValued` | `orig=[0,1,2,4]`, `target=[0,1,2,3]` | Remaps labels to contiguous indices |
| `ConcatItemsd` | `name="image"` | Merges 4 modalities into a 4-channel tensor |
| `NormalizeIntensityd` | `nonzero=True, channel_wise=True` | Z-score normalisation per modality, excluding skull-stripped zeros |
| `SpatialPadd` | `spatial_size=(128,128,64)` | Pads volumes below the minimum ROI size |
| `RandSpatialCropd`* | `roi_size=(128,128,64)` | Uniform patch extraction for batched training |
| `RandFlipd`* | `prob=0.5, spatial_axis=[0,1,2]` | Data augmentation — random axis flips |
| `RandRotate90d`* | `prob=0.5, max_k=3` | Data augmentation — 90° rotation variants |
| `ToTensord` | `keys=["image", "label"]` | Converts arrays to PyTorch tensors |

*\* Training split only — validation/test use only the deterministic base transforms.*

Intensity normalisation with `nonzero=True` and `channel_wise=True` preserves each modality's distinct contrast profile while removing scanner-dependent scale differences — important given BraTS's multi-institutional acquisition.

---

## 🏋️ Training Configuration

| Setting | Value |
| :--- | :--- |
| Optimiser | Adam, `lr=1e-4`, `weight_decay=1e-5` |
| Loss | `DiceCELoss` (see below) |
| Precision | Mixed precision (`torch.amp.autocast` + `GradScaler`) |
| Train batch size | 2 (`shuffle=True`, `num_workers=4`) |
| Val/Test batch size | 1 (deterministic, no shuffling) |
| Epochs | 15 |
| Inference | `sliding_window_inference`, ROI `(128,128,64)`, `overlap=0.5`, `sw_batch_size=2` |
| Metric | `DiceMetric` (`include_background=True`, mean reduction) |

At each training step: gradients are cleared, the forward pass runs under autocast, `DiceCELoss` is computed, scaled gradients are back-propagated, and weights are updated. Validation runs each epoch with gradients disabled to conserve memory.

---

## 📉 Loss Function

The models are optimised using **DiceCELoss**, a hybrid function that handles extreme class imbalance (background dominates >99% of voxels) while maintaining voxel-wise classification pressure.

1. **Dice Loss (regional overlap):**
   $$L_{Dice} = 1 - \frac{2\sum_{i}p_{i}g_{i}+\epsilon}{\sum_{i}p_{i}+\sum_{i}g_{i}+\epsilon}$$
2. **Cross-Entropy Loss (voxel-wise classification):**
   $$L_{CE} = -\sum_{i}g_{i}\log(p_{i})$$
3. **Total Loss:**
   $$L_{Total} = L_{Dice} + \lambda L_{CE}$$

*(Where $p_i$ is the predicted probability, $g_i$ is the ground-truth label, and $\epsilon$ is a smoothing constant, `1e-5`.)*

---

## 📊 Quantitative Results

Models were evaluated on a held-out test set of 126 cases.

| Architecture | Training Loss | Validation Loss | Validation Dice | Test Loss | Test Dice |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **U-Net (Baseline)** | 0.4242 | 0.4271 | 0.6748 | 0.3358 | 0.7613 |
| **ResUNet** | **0.3666** | 0.4249 | 0.6556 | **0.2509** | 0.8133 |
| **DynUNet** | 0.4106 | 0.4871 | 0.6665 | 0.3712 | 0.8204 |
| **Attention U-Net** | 0.3840 | **0.4307** | **0.6798** | 0.2914 | **0.8418** |

> *Test Dice is the aggregate Dice score across all foreground classes (`include_background=True`). Per-region (WT/TC/ET) breakdowns and HD95 boundary metrics were part of the intended evaluation protocol but are not reported here.*

**Attention U-Net** achieved the best overall performance — highest validation Dice (0.6798) and test Dice (0.8418) — consistent with its architectural motivation of suppressing irrelevant background while amplifying tumour-relevant regions.

**ResUNet** attained the lowest training loss (0.3666) and strong test Dice (0.8133), but a lower validation Dice than the baseline, illustrating that lower training loss doesn't guarantee better overlap-based generalisation.

**DynUNet** posted a competitive test Dice (0.8204) despite the highest validation loss (0.4871), suggesting it captures multi-scale context effectively but with less consistent validation behaviour.

The **baseline U-Net**, while stable during training, produced the lowest test Dice (0.7613) — evidence that architectural enhancements (residual connections, attention, dynamic kernels) meaningfully improve fine-grained boundary segmentation over the vanilla architecture.

For context, published BraTS results for comparable-capacity models report whole-tumour Dice around 0.84–0.89 (2018–2020 3D U-Nets / nnU-Net), rising to 0.926 for SwinUNETR (2021) — the gap here reflects this project's smaller channel widths and shorter (15-epoch) training schedule rather than a fundamental architectural shortcoming.

---

## 🖼️ Qualitative Results

Predicted segmentation masks are overlaid on the corresponding MRI slices for randomly sampled validation cases, alongside the FLAIR input and ground truth mask. Inspection reveals:

- **Strengths:** accurate localisation of the whole tumour mass and correct identification of the oedema extent.
- **Failure modes:** fragmented enhancing-tumour predictions and underestimation of necrotic core boundaries — both actionable targets for future model improvements.

---

## ⚠️ Limitations

- **Unstratified split:** the train/val/test partition is positional, not stratified by tumour grade or histological subtype, which may introduce subtle evaluation bias.
- **Patch-based training:** 128×128×64 patches are computationally necessary but may miss global spatial context available in the full brain volume.
- **Constrained model capacity:** the `(16,32,64,128,256)` channel configuration is smaller than state-of-the-art BraTS setups (often `32–512`), potentially limiting performance on the harder TC and ET sub-regions.

## 🔮 Future Directions

- Increase model capacity and enable full-volume sliding-window inference at evaluation time to reduce prediction fragmentation.
- Integrate transformer-based encoders (e.g. SwinUNETR-style Swin Transformer blocks) to capture long-range 3D dependencies.
- Apply Explainable AI techniques (Grad-CAM, attention map visualisation) for clinically meaningful interpretability.
- Explore post-training quantisation for real-time inference on standard hospital workstations.

---

## 💻 Streamlit Web Application (`app.py`)

An interactive dashboard demonstrating the best-performing model (Attention U-Net) in real time.

### How it Works
1. **Model instantiation:** provisions the `AttentionUnet` architecture and loads pre-trained weights from `attention_unet_brats.pth` onto CPU.
2. **Volumetric inference:** uses MONAI's `sliding_window_inference` to extract 128×128×64 ROIs on-the-fly and blend overlapping patches into a full-resolution prediction volume, avoiding OOM errors on large 3D brains.
3. **Data discretisation:** `AsDiscrete(argmax=True)` converts continuous probability logits into a definitive per-voxel class label.
4. **Colour-coded visualisation:** a custom `matplotlib.colors.ListedColormap` maps segmentation classes to the standardised clinical protocol:
   - ⬛ **Background:** `(0.05, 0.05, 0.05)`
   - 🟥 **Necrotic Core:** `(0.90, 0.10, 0.10)`
   - 🟩 **Peritumoral Edema:** `(0.10, 0.80, 0.10)`
   - 🟦 **Enhancing Tumor:** `(0.10, 0.45, 0.95)`
5. **Interactive UI:** masks are plotted over the original MRI slices, letting users scroll through the Z-axis to visually validate predictions.

---

## 🚀 Getting Started

### Prerequisites
Python 3.8+ is required. Use a virtual environment (`venv` or `conda`).

```bash
git clone https://github.com/singhvi28/bt-seg.git
cd bt-seg
```

### Installation
```bash
pip install torch torchvision torchaudio
pip install monai streamlit matplotlib pillow numpy
```

### Download the Pre-trained Weights
The Attention U-Net weights (`attention_unet_brats.pth`) are hosted on Google Drive:

🔗 **[Download attention_unet_brats.pth](https://drive.google.com/file/d/1TPMOYYi2DvhkqEJuMMVkMnp2BcP4eniV/view?usp=sharing)**

Either download it manually from the link above and place it in the project root, or fetch it directly with `gdown`:
```bash
pip install gdown
gdown --id 1TPMOYYi2DvhkqEJuMMVkMnp2BcP4eniV -O attention_unet_brats.pth
```

### Launching the App
1. Confirm `attention_unet_brats.pth` is in the project root directory.
2. Run the Streamlit server:
```bash
streamlit run app.py
```
3. Open the local network URL shown in the terminal (usually `http://localhost:8501`).

---

## 📖 References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, LNCS 9351.
- Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Zhou, Z. et al. (2019). *UNet++: A Nested U-Net Architecture for Medical Image Segmentation.* DLMIA/ML-CDS, LNCS 11045.
- Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas.* MIDL. arXiv:1804.03999.
- Isensee, F. et al. (2021). *nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation.* Nature Methods, 18(2), 203–211.
- Hatamizadeh, A. et al. (2022). *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images.* MICCAI Brainlesion Workshop, LNCS 12962.
- Bakas, S. et al. (2021). *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification.* arXiv:2107.02314.
- Cardoso, M. J. et al. (2022). *MONAI: An Open-Source Framework for Deep Learning in Healthcare.* arXiv:2211.02701.

---

## 📄 License
This project is released under the MIT License. See `LICENSE` for details.