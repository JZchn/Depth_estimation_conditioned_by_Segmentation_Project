"""
run_eval_metrics.py
Compute batch statistics (RMSE, MAE, AbsRel, wRMSE) for Raw vs Guided depth
across 50 real endoscopic images, with paired t-test significance.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

from src.config import USE_REAL_LOCAL_DATA
USE_REAL_LOCAL_DATA = True
import importlib
import src.config
importlib.reload(src.config)

from src.config import base_path, device, image_dir, gt_depth_dir, test_mask_dir
from src.sam_segmentation import load_sam, generate_mask_with_gui
from src.reliability import compute_reliability_and_features, apply_reliability_to_image
from src.depth_inference import load_depth_model, infer_depth
from src.metrics import compute_depth_metrics, compute_batch_statistics, print_metrics_table


def main():
    # Load checkpoints
    sam_checkpoint = os.path.join(base_path, "checkpoints", "sam_vit_b_01ec64.pth")
    da_checkpoint = os.path.join(base_path, "checkpoints", "depth_anything_v2_metric_hypersim_vitb.pth")

    print(">>> Loading SAM...")
    sam_predictor = load_sam(sam_checkpoint)
    print(">>> Loading Depth Anything V2...")
    depth_model = load_depth_model(da_checkpoint)

    # Read all images
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    print(f">>> {len(all_files)} images found in {image_dir}.")
    sample_files = all_files[:50]

    raw_metrics_list = []
    guided_metrics_list = []

    for fname in tqdm(sample_files):
        img_path = os.path.join(image_dir, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # Load or generate SAM mask
        mask_path = os.path.join(test_mask_dir, fname)
        points_path = mask_path.replace(".png", "_points.npy")

        if os.path.exists(mask_path) and os.path.exists(points_path):
            mask_sam = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            points = np.load(points_path)
        else:
            print(f"[{fname}] Generating SAM mask with GUI...")
            mask_sam, points = generate_mask_with_gui(img_path, sam_predictor, save_path=mask_path)
            if mask_sam is None:
                print(f"  Skipped {fname} (user cancelled or insufficient points)")
                continue
            np.save(points_path, points)

        # Load ground truth depth
        gt_path = os.path.join(gt_depth_dir, fname.replace(".png", ".tiff"))
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            print(f"  Skipped {fname} (GT not found: {gt_path})")
            continue
        depth_gt = gt.astype(np.float32)
        if depth_gt.max() > 50:
            depth_gt /= 1000.0

        # Compute reliability map
        reliability_map, mask_float, features = compute_reliability_and_features(img_bgr, mask_path)

        # Depth inference: Raw and Guided
        depth_raw = infer_depth(depth_model, img_bgr)
        img_weighted = apply_reliability_to_image(img_bgr, reliability_map)
        depth_guided = infer_depth(depth_model, img_weighted)

        # Define valid mask: tool mask region where GT is valid
        valid_mask = (mask_sam > 0) & (depth_gt > 0) & np.isfinite(depth_gt)

        if valid_mask.sum() < 100:
            print(f"  Skipped {fname} (too few valid pixels: {valid_mask.sum()})")
            continue

        # Compute per-image metrics
        raw_metrics = compute_depth_metrics(depth_raw, depth_gt, valid_mask, reliability=reliability_map)
        guided_metrics = compute_depth_metrics(depth_guided, depth_gt, valid_mask, reliability=reliability_map)

        raw_metrics_list.append(raw_metrics)
        guided_metrics_list.append(guided_metrics)

    # Aggregate batch statistics
    N = len(raw_metrics_list)
    print(f"\n>>> Computed metrics for {N} images (out of {len(sample_files)} total).")

    if N < 2:
        print("ERROR: Not enough samples for batch statistics. Exiting.")
        return

    stats = compute_batch_statistics(raw_metrics_list, guided_metrics_list)
    print_metrics_table(stats)

    # Save results to CSV for later reference
    import csv
    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Raw_Mean", "Raw_Std", "Guided_Mean", "Guided_Std", "P_value", "Significance"])
        for key in ["RMSE", "MAE", "AbsRel", "wRMSE", "wAbsRel"]:
            if key not in stats:
                continue
            s = stats[key]
            sig = ""
            if s["p_value"] < 0.001:
                sig = "***"
            elif s["p_value"] < 0.01:
                sig = "**"
            elif s["p_value"] < 0.05:
                sig = "*"
            writer.writerow([key, f"{s['raw_mean']:.4f}", f"{s['raw_std']:.4f}",
                             f"{s['guided_mean']:.4f}", f"{s['guided_std']:.4f}",
                             f"{s['p_value']:.4f}", sig])
    print(f">>> CSV summary saved to {csv_path}")

    # Also save per-image metrics for detailed analysis
    per_image_path = os.path.join("outputs", "metrics_per_image.csv")
    with open(per_image_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Collect all possible metric keys
        all_keys = list(raw_metrics_list[0].keys())
        writer.writerow(["Image"] + [f"Raw_{k}" for k in all_keys] + [f"Guided_{k}" for k in all_keys])
        for i, (rm, gm) in enumerate(zip(raw_metrics_list, guided_metrics_list)):
            row = [sample_files[i]]
            for k in all_keys:
                row.append(f"{rm[k]:.4f}")
            for k in all_keys:
                row.append(f"{gm[k]:.4f}")
            writer.writerow(row)
    print(f">>> Per-image metrics saved to {per_image_path}")


if __name__ == "__main__":
    main()