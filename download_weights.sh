#!/bin/bash

# 创建 checkpoints 文件夹（如果不存在）
mkdir -p checkpoints

echo "======================================================="
echo "Downloading SAM weights (sam_vit_b_01ec64.pth)..."
echo "======================================================="
wget -O checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

echo ""
echo "======================================================="
echo "Downloading Depth Anything V2 weights (depth_anything_v2_metric_hypersim_vitb.pth)..."
echo "======================================================="
wget -O checkpoints/depth_anything_v2_metric_hypersim_vitb.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-ViT-B/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth

echo ""
echo "======================================================="
echo "All weights downloaded successfully!"
echo "======================================================="