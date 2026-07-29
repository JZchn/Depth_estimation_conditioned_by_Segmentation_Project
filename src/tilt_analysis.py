import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.metrics import sample_depth_robust

def analyze_tool_tilt_from_pred(points, depth_map, eps=1e-3):
    """Analyze tool tilt direction from depth map."""
    depths = []
    for (x, y) in points:
        z = sample_depth_robust(depth_map, int(x), int(y), r=3)
        depths.append(z)

    d_tip, d_mid, d_end = depths

    if np.any(np.isnan(depths)):
        return {"valid": False, "reason": "invalid_depth_sample", "depths": depths}

    increasing = (d_tip + eps < d_mid) and (d_mid + eps < d_end)
    decreasing = (d_tip > d_mid + eps) and (d_mid > d_end + eps)

    if increasing:
        return {"valid": True, "is_tilted": True, "direction": "toward_camera_entry (tip shallower)", "depths": depths}
    if decreasing:
        return {"valid": True, "is_tilted": True, "direction": "toward_body (tip deeper)", "depths": depths}

    return {"valid": True, "is_tilted": False, "direction": "approximately_parallel_or_uncertain", "depths": depths}

def _draw_tool_tilt_on_ax(ax, img_bgr, points, tilt_info, title_suffix=""):
    """Internal function: draw tool tilt on given axis."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    colors = ["red", "yellow", "blue"]
    labels = ["Tip", "Mid", "End"]

    for (x, y), c, l, d in zip(points, colors, labels, tilt_info["depths"]):
        ax.scatter(x, y, c=c, s=60, zorder=3)
        ax.text(x + 5, y - 5, f"{l}\nz={d:.3f}", color=c, fontsize=8,
                bbox=dict(facecolor="black", alpha=0.4, pad=1))

    ax.plot(points[:, 0], points[:, 1], "--", color="cyan", linewidth=1.5, zorder=2)

    title = f"{title_suffix}\n{tilt_info['direction']}" if tilt_info["valid"] else f"{title_suffix}\nInvalid"
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def visualize_tool_tilt_comparison(raw_img, guided_img, points, tilt_raw, tilt_guided, save_path=None):
    """Plot comparison figure: Raw vs Guided tilt analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=200)
    _draw_tool_tilt_on_ax(axes[0], raw_img, points, tilt_raw, title_suffix="Raw")
    _draw_tool_tilt_on_ax(axes[1], guided_img, points, tilt_guided, title_suffix="Guided")
    plt.tight_layout(pad=0.2)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()