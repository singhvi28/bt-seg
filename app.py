import io
import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from monai.networks.nets import AttentionUnet
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "attention_unet_brats.pth")
DEVICE     = torch.device("cpu")
ROI        = (128, 128, 64)
IMG_SIZE   = (128, 128)
N_DEPTH    = 64

LABEL_META = {
    0: {"name": "Background",        "color": (0.05, 0.05, 0.05)},
    1: {"name": "Necrotic Core",     "color": (0.90, 0.10, 0.10)},
    2: {"name": "Peritumoral Edema", "color": (0.10, 0.80, 0.10)},
    3: {"name": "Enhancing Tumor",   "color": (0.10, 0.45, 0.95)},
}
SEG_CMAP = mcolors.ListedColormap([m["color"] for m in LABEL_META.values()])

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}\n"
            "Place 'attention_unet_brats.pth' in the same folder as app.py."
        )
    net = AttentionUnet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


# ── Pre-processing ────────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> torch.Tensor:
    """
    PNG -> (1, 4, 128, 128, 64) tensor.

    - Grayscale + resize to 128x128.
    - Nonzero z-score normalisation (mirrors NormalizeIntensityd nonzero=True).
    - Replicated to 4 channels (flair / t1 / t1ce / t2).
    - Stacked 64x along depth axis (pseudo-3-D volume).
    """
    gray = np.array(img.convert("L").resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32)
    mask = gray > 0
    if mask.sum() > 0:
        gray[mask] = (gray[mask] - gray[mask].mean()) / (gray[mask].std() + 1e-8)

    vol  = np.stack([gray] * N_DEPTH, axis=-1)   # (128, 128, 64)
    vol4 = np.stack([vol]  * 4,       axis=0)    # (4,  128, 128, 64)
    return torch.from_numpy(vol4).unsqueeze(0)    # (1,  4,  128, 128, 64)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(net, tensor: torch.Tensor) -> np.ndarray:
    """Returns (128, 128, 64) int32 class-index array."""
    post = AsDiscrete(argmax=True)
    with torch.no_grad():
        out  = sliding_window_inference(
            tensor, ROI, sw_batch_size=1,
            predictor=net, overlap=0.5, mode="gaussian",
        )
        pred = post(out[0])                               # (1, 128, 128, 64)
    return pred.squeeze(0).cpu().numpy().astype(np.int32) # (128, 128, 64)


# ── Visualisation ─────────────────────────────────────────────────────────────
def make_figure(original: Image.Image, seg_vol: np.ndarray, slice_idx: int):
    gray   = np.array(
        original.convert("L").resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32
    )
    seg    = seg_vol[:, :, slice_idx]
    g_norm = gray / (gray.max() + 1e-8)

    # coloured overlay
    overlay = np.stack([g_norm] * 3, axis=-1)
    alpha   = 0.55
    for lbl, meta in LABEL_META.items():
        if lbl == 0:
            continue
        mask = seg == lbl
        if mask.any():
            for c, v in enumerate(meta["color"]):
                overlay[:, :, c] = np.where(
                    mask,
                    overlay[:, :, c] * (1 - alpha) + v * alpha,
                    overlay[:, :, c],
                )
    overlay = np.clip(overlay, 0, 1)

    bg  = "#0e1117"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                             facecolor=bg, constrained_layout=True)
    panels = [
        ("Original MRI",      gray,    {"cmap": "gray"}),
        ("Segmentation Mask", seg,     {"cmap": SEG_CMAP, "vmin": 0, "vmax": 3,
                                        "interpolation": "nearest"}),
        ("Overlay",           overlay, {}),
    ]
    for ax, (title, data, kw) in zip(axes, panels):
        ax.set_facecolor(bg)
        ax.imshow(data, **kw)
        ax.set_title(title, color="white", fontsize=13, pad=8)
        ax.axis("off")

    patches = [
        mpatches.Patch(facecolor=m["color"], label=m["name"],
                       edgecolor="white", linewidth=0.4)
        for lbl, m in LABEL_META.items() if lbl != 0
    ]
    axes[1].legend(handles=patches, loc="lower right", fontsize=8,
                   framealpha=0.65, labelcolor="white", facecolor="#1a1a2e")
    return fig, seg


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="BraTS Segmentation", page_icon="🧠", layout="wide")

st.markdown(
    "<h1 style='text-align:center;color:#4fc3f7;'>🧠 Brain Tumor Segmentation</h1>"
    "<p style='text-align:center;color:#aaa;font-size:15px;'>"
    "Attention U-Net · BraTS 2021 · 4 classes</p>"
    "<hr style='border:0;border-top:1px solid #333;'>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    slice_idx = st.slider(
        "Depth slice to display",
        min_value=0, max_value=N_DEPTH - 1, value=N_DEPTH // 2,
        help="The model predicts across 64 depth levels. Slide to inspect each.",
    )
    st.markdown("---")
    st.markdown("""
**How it works**

1. PNG → grayscale, resized to 128 × 128.
2. Replicated to **4 channels** and **64 depth slices** to form a pseudo-3-D volume.
3. Attention U-Net runs sliding-window inference.
4. The chosen depth slice is rendered.

> ⚠️ For accurate clinical results supply real multi-modal NIfTI volumes.
""")

uploaded = st.file_uploader(
    "Upload a brain MRI image (axial slice recommended)",
    type=["png", "jpg", "jpeg"],
)
if not uploaded:
    st.info("👆  Upload an MRI image to begin.")
    st.stop()

image = Image.open(uploaded)

try:
    net = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

c1, c2 = st.columns([1, 2])
with c1:
    st.image(image, caption="Uploaded image", use_container_width=True)
with c2:
    st.markdown("**Image info**")
    st.write(f"- Original size : {image.size[0]} × {image.size[1]} px")
    st.write(f"- Colour mode   : {image.mode}")
    st.write(f"- Model input   : 128 × 128 × {N_DEPTH}  ×  4 channels")

with st.spinner("Running inference — this may take ~30–60 s on CPU…"):
    tensor  = preprocess(image)
    seg_vol = run_inference(net, tensor)

st.success("✅ Inference complete!")
st.markdown("---")

fig, seg_slice = make_figure(image, seg_vol, slice_idx)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Per-slice stats
st.markdown("### Segmentation statistics  (selected slice)")
total     = seg_slice.size
stat_cols = st.columns(4)
for col, (lbl, meta) in zip(stat_cols, LABEL_META.items()):
    cnt = int((seg_slice == lbl).sum())
    col.metric(label=meta["name"], value=f"{cnt / total * 100:.1f} %",
               delta=f"{cnt} px", delta_color="off")

# Full-volume stats
with st.expander("📊 Full-volume class distribution"):
    import pandas as pd
    uniq, cnts = np.unique(seg_vol, return_counts=True)
    vol_total  = seg_vol.size
    st.dataframe(
        pd.DataFrame([
            {
                "Class":      LABEL_META[int(lbl)]["name"],
                "Voxels":     int(cnt),
                "Percentage": f"{cnt / vol_total * 100:.2f} %",
            }
            for lbl, cnt in zip(uniq, cnts)
        ]),
        use_container_width=True,
        hide_index=True,
    )

# Download segmentation volume
buf = io.BytesIO()
np.save(buf, seg_vol.astype(np.int8))
st.download_button(
    label="⬇️ Download segmentation volume (.npy)",
    data=buf.getvalue(),
    file_name="segmentation.npy",
    mime="application/octet-stream",
)