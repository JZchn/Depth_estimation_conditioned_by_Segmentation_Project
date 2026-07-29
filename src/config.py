import os
import torch
import numpy as np
import matplotlib
import cv2
import sys

# --- Configuration Switches ---
# USE_REAL_LOCAL_DATA = True  -> use local endoscopic data (50 images)
# USE_REAL_LOCAL_DATA = False -> use synthetic data for GitHub demo
USE_REAL_LOCAL_DATA = True
# -----------------------------

matplotlib.use("Agg")

base_path = os.getcwd()

# Data paths based on configuration switch
if USE_REAL_LOCAL_DATA:
    image_dir = os.path.join(base_path, "data", "rect_left")
    gt_depth_dir = os.path.join(base_path, "data", "GT_left", "depth_maps")
    print(f">>> Loading local real data from: {image_dir}")
else:
    image_dir = os.path.join(base_path, "data", "sample")
    gt_depth_dir = os.path.join(base_path, "data", "sample")
    print(">>> Loading synthetic demo data")

# Output directories
output_mask_dir = os.path.join(base_path, "outputs", "sam_masks")
test_mask_dir = os.path.join(base_path, "outputs", "test", "sam_mask")
figures_output_dir = os.path.join(base_path, "outputs", "figures")
demo_figures_dir = os.path.join(base_path, "outputs", "demo_figures")

# Auto-create all output directories
for d in [output_mask_dir, test_mask_dir, figures_output_dir, demo_figures_dir]:
    os.makedirs(d, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
np.random.seed(20)
