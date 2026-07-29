# run_local_eval.py
# manually select 3 points on the image

import os
import cv2
import numpy as np
from tqdm import tqdm

from src.config import USE_REAL_LOCAL_DATA
USE_REAL_LOCAL_DATA = True
import importlib
import src.config
importlib.reload(src.config)

from src.config import base_path, device, image_dir, gt_depth_dir, output_mask_dir, figures_output_dir, test_mask_dir
from src.sam_segmentation import load_sam, generate_mask_with_gui
from src.reliability import compute_reliability_and_features, apply_reliability_to_image
from src.depth_inference import load_depth_model, infer_depth
from src.tilt_analysis import analyze_tool_tilt_from_pred, visualize_tool_tilt_comparison
from src.visualization import visualize_features_paper, visualize_guidance_figure_A, visualize_depth_comparison_figure_B, visualize_comprehensive_appendix

def main():
    # Load checkpoints
    sam_checkpoint = os.path.join(base_path, "checkpoints", "sam_vit_b_01ec64.pth")
    da_checkpoint = os.path.join(base_path, "checkpoints", "depth_anything_v2_metric_hypersim_vitb.pth")
    
    print(">>> loading SAM...")
    sam_predictor = load_sam(sam_checkpoint)
    print(">>> loading Depth Anything V2...")
    depth_model = load_depth_model(da_checkpoint)

    # read all images in the directory
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    print(f">>>  {len(all_files)} images found in {image_dir}.")

    sample_files = all_files[:50]  # process 50 test images

    for fname in tqdm(sample_files):
        img_path = os.path.join(image_dir, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue

        # Check if SAM mask and points already exist
        mask_path = os.path.join(test_mask_dir, fname)      
        points_path = mask_path.replace(".png", "_points.npy")
        
        if os.path.exists(mask_path) and os.path.exists(points_path):
            print(f"[{fname}] SAM mask and points already exist. Loading them...")
            mask_sam = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            points = np.load(points_path)
        else:
            print(f"[{fname}] Generating SAM mask with GUI...")
            mask_sam, points = generate_mask_with_gui(img_path, sam_predictor, save_path=mask_path)
            if mask_sam is None:
                print(f"user cancelled {fname} ")
                continue
            # Save the clicked points
            np.save(points_path, points)
            print(f"save points coordinates to {points_path}")

        # Load ground truth depth map
        gt_path = os.path.join(gt_depth_dir, fname.replace(".png", ".tiff"))
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None: continue
        depth_gt = gt.astype(np.float32)
        if depth_gt.max() > 50: depth_gt /= 1000.0

        # Compute reliability map and features
        reliability_map, mask_float, features = compute_reliability_and_features(img_bgr, mask_path)

        # Depth inference (Raw and Guided)
        depth_raw = infer_depth(depth_model, img_bgr)
        img_weighted = apply_reliability_to_image(img_bgr, reliability_map)
        depth_guided = infer_depth(depth_model, img_weighted)

        # Tilt analysis (for demonstration)
        tilt_raw = analyze_tool_tilt_from_pred(points, depth_raw)
        tilt_guided = analyze_tool_tilt_from_pred(points, depth_guided)
        tilt_save_path = os.path.join(figures_output_dir, fname.replace(".png", "_tilt_compare.pdf"))
        visualize_tool_tilt_comparison(img_bgr, img_weighted, points, tilt_raw, tilt_guided, save_path=tilt_save_path)

        # Generate all academic figures to outputs/figures/
        save_path_base = os.path.join(figures_output_dir, fname.replace(".png", ""))
        print(f">>> saving {fname} results to outputs/figures/ ...")
        visualize_features_paper(img_bgr, features, save_path=save_path_base + "_features.pdf")
        visualize_guidance_figure_A(img_bgr, reliability_map, save_path=save_path_base + "_guidance.pdf")
        visualize_depth_comparison_figure_B(depth_gt, depth_raw, depth_guided, save_path=save_path_base + "_depth_compare.pdf")
        visualize_comprehensive_appendix(img_bgr, depth_raw, depth_guided, depth_gt, reliability_map, mask_sam, save_path=save_path_base + "_appendix.pdf")

    print("done, check outputs/figures/ for the generated PDF files.")

if __name__ == "__main__":
    main()