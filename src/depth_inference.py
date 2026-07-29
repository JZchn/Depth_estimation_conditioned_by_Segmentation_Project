import sys
import os
import torch
from src.config import base_path, device

# add Depth-Anything-V2 source path
depth_anything_path = os.path.join(base_path, "Depth-Anything-V2", "metric_depth")
if os.path.exists(depth_anything_path):
    sys.path.append(depth_anything_path)
else:
    print("[WARNING] Depth-Anything-V2 not found in root, please clone it.")

def load_depth_model(checkpoint_path):
    """Load Depth Anything V2 model."""
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        raise ImportError("Please ensure 'Depth-Anything-V2' is installed correctly.")

    model = DepthAnythingV2(
        encoder='vitb',
        features=128,
        out_channels=[96, 192, 384, 768]
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.to(device).eval()
    return model

def infer_depth(model, img_bgr):
    """Run depth inference on a single image."""
    return model.infer_image(img_bgr)