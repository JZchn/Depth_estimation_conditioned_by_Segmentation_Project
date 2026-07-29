import os
import cv2
import numpy as np
from src.config import base_path, device, output_mask_dir, demo_figures_dir
from src.sam_segmentation import load_sam, generate_mask_with_points
from src.reliability import compute_reliability_and_features, apply_reliability_to_image
from src.depth_inference import load_depth_model, infer_depth
from src.metrics import compute_depth_metrics
from src.tilt_analysis import analyze_tool_tilt_from_pred, visualize_tool_tilt_comparison
from src.visualization import visualize_features_paper, visualize_guidance_figure_A, visualize_depth_comparison_figure_B, visualize_comprehensive_appendix

def generate_dummy_image():
    """Generate a synthetic dummy image to replace private endoscopic data for demo."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 50)
    cv2.ellipse(img, (320, 200), (40, 120), 30, 0, 360, (180, 180, 180), -1)
    cv2.circle(img, (310, 180), 6, (255, 255, 255), -1)
    return img

def main():
    # 1. Load checkpoints
    sam_checkpoint = os.path.join(base_path, "checkpoints", "sam_vit_b_01ec64.pth")
    da_checkpoint = os.path.join(base_path, "checkpoints", "depth_anything_v2_metric_hypersim_vitb.pth")
    
    print(">>> Loading SAM...")
    sam_predictor = load_sam(sam_checkpoint)
    print(">>> Loading Depth Anything V2...")
    depth_model = load_depth_model(da_checkpoint)

    # 2. Generate dummy input (privacy: no real endoscopic images used)
    print(">>> Generating dummy input...")
    img_bgr = generate_dummy_image()
    fname = "demo.png"
    
    # 3. Generate SAM mask with hardcoded points (skip GUI)
    mask_path = os.path.join(output_mask_dir, fname)
    # Hardcoded 3 points: Top (Tip), Middle, Bottom (End)
    hardcoded_points = [[320, 100], [320, 250], [320, 400]]
    mask_sam, points = generate_mask_with_points(img_bgr, sam_predictor, points_override=hardcoded_points, save_path=mask_path)

    # 4. Compute features and reliability map
    reliability_map, mask_float, features = compute_reliability_and_features(img_bgr, mask_path)

    # 5. Depth estimation (Raw vs Guided)
    depth_raw = infer_depth(depth_model, img_bgr)
    img_weighted = apply_reliability_to_image(img_bgr, reliability_map)
    depth_guided = infer_depth(depth_model, img_weighted)

    # 6. Tilt analysis (demo only)
    tilt_raw = analyze_tool_tilt_from_pred(points, depth_raw)
    tilt_guided = analyze_tool_tilt_from_pred(points, depth_guided)
    tilt_save_path = os.path.join(demo_figures_dir, "tool_tilt_compare.pdf")
    visualize_tool_tilt_comparison(img_bgr, img_weighted, points, tilt_raw, tilt_guided, save_path=tilt_save_path)
    
    # 7. Generate all academic figures to outputs/demo_figures/
    print(">>> Saving results to outputs/demo_figures/ ...")
    visualize_features_paper(img_bgr, features, save_path=os.path.join(demo_figures_dir, "features.pdf"))
    visualize_guidance_figure_A(img_bgr, reliability_map, save_path=os.path.join(demo_figures_dir, "guidance_A.pdf"))
    # Since there is no real ground truth, use dummy depth for demo
    visualize_depth_comparison_figure_B(depth_raw, depth_raw, depth_guided, save_path=os.path.join(demo_figures_dir, "depth_B.pdf"))
    visualize_comprehensive_appendix(img_bgr, depth_raw, depth_guided, depth_raw, reliability_map, mask_sam, save_path=os.path.join(demo_figures_dir, "appendix.pdf"))

    print("Demo completed! Check outputs/demo_figures/ for PDF files.")

if __name__ == "__main__":
    main()