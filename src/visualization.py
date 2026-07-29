import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.reliability import apply_reliability_to_image

def _load_img(path_or_array):
    """Unified image loading interface, supports file path or numpy array."""
    if isinstance(path_or_array, str):
        return cv2.cvtColor(cv2.imread(path_or_array), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(path_or_array, cv2.COLOR_BGR2RGB) if path_or_array.shape[-1] == 3 else path_or_array

def visualize_features_paper(img_path, features, save_path=None):
    """Plot feature extraction figure (RGB + Shading + Texture + Edge + Specular)."""
    img = _load_img(img_path)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.2), dpi=300, gridspec_kw={"wspace": 0.02})
    axes[0].imshow(img)
    axes[0].set_title("RGB", fontsize=9)
    axes[0].axis("off")

    feature_order = ["Shading", "Texture", "Edge", "Specular"]
    for ax, name in zip(axes[1:], feature_order):
        feat = features[name]
        ax.imshow(feat, cmap="gray" if name != "Specular" else "gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    plt.tight_layout(pad=0.2)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def visualize_guidance_figure_A(img_path, reliability, save_path):
    """Plot Guidance comparison figure (RGB -> Reliability -> Guided)."""
    img = _load_img(img_path)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_weighted = apply_reliability_to_image(img_bgr, reliability)
    img_weighted = cv2.cvtColor(img_weighted, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=300, gridspec_kw={"wspace": 0.02})
    axes[0].imshow(img); axes[0].set_title("(a) RGB", fontsize=9)
    axes[1].imshow(reliability, cmap="magma", vmin=0, vmax=1); axes[1].set_title("(b) Reliability", fontsize=9)
    axes[2].imshow(img_weighted); axes[2].set_title("(c) Guided Input", fontsize=9)
    for ax in axes: ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()

def visualize_depth_comparison_figure_B(depth_gt, depth_raw, depth_guided, save_path):
    """Plot depth comparison figure (GT -> Raw -> Guided)."""
    def norm(d):
        d = np.nan_to_num(d)
        return (d - d.min()) / (d.max() - d.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=300, gridspec_kw={"wspace": 0.02})
    axes[0].imshow(norm(depth_gt), cmap="inferno"); axes[0].set_title("(a) GT", fontsize=9)
    axes[1].imshow(norm(depth_raw), cmap="inferno"); axes[1].set_title("(b) Raw", fontsize=9)
    axes[2].imshow(norm(depth_guided), cmap="inferno"); axes[2].set_title("(c) Guided", fontsize=9)
    for ax in axes: ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()

def visualize_comprehensive_appendix(img_path, depth_raw, depth_guided, depth_gt, reliability, mask, save_path):
    """Plot comprehensive appendix figure (2x5 grid)."""
    img = _load_img(img_path)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_weighted = apply_reliability_to_image(img_bgr, reliability)
    img_weighted = cv2.cvtColor(img_weighted, cv2.COLOR_BGR2RGB)

    def norm(d):
        d = np.nan_to_num(d)
        return (d - d.min()) / (d.max() - d.min() + 1e-8)

    err_raw = np.abs(depth_raw - depth_gt)
    err_raw[depth_gt <= 0] = 0
    err_weighted = err_raw * reliability

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), dpi=300)
    axes[0,0].imshow(img); axes[0,0].set_title("RGB")
    axes[0,1].imshow(mask, cmap="gray"); axes[0,1].set_title("Mask")
    axes[0,2].imshow(img_weighted); axes[0,2].set_title("Guided Input")
    axes[0,3].imshow(norm(depth_gt), cmap="inferno"); axes[0,3].set_title("Ground Truth")
    axes[0,4].imshow(reliability, cmap="magma"); axes[0,4].set_title("Reliability")

    axes[1,0].imshow(norm(depth_raw), cmap="inferno"); axes[1,0].set_title("Raw")
    axes[1,1].imshow(norm(depth_guided), cmap="inferno"); axes[1,1].set_title("Guided")
    axes[1,2].imshow(norm(err_raw), cmap="jet"); axes[1,2].set_title("Error")
    axes[1,3].imshow(norm(err_weighted), cmap="jet"); axes[1,3].set_title("Weighted Error")
    axes[1,4].axis("off")

    for ax in axes.flatten(): ax.axis("off")
    plt.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()