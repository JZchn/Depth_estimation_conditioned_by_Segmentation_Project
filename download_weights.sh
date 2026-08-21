#!/bin/bash

# 创建 checkpoints 文件夹（如果不存在）
mkdir -p checkpoints

# Download helper: prefer wget, fall back to curl (curl is preinstalled on
# Windows 10+/11 and Git for Windows).
if command -v wget >/dev/null 2>&1; then
    DL() { wget -O "$1" "$2"; }
else
    DL() { curl -L -o "$1" "$2"; }
fi

echo "======================================================="
echo "Downloading SAM weights (sam_vit_b_01ec64.pth)..."
echo "======================================================="
DL checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

echo ""
echo "======================================================="
echo "Downloading Depth Anything V2 weights (depth_anything_v2_metric_hypersim_vitb.pth)..."
echo "======================================================="
DL checkpoints/depth_anything_v2_metric_hypersim_vitb.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth

echo ""
echo "======================================================="
echo "All weights downloaded successfully!"
echo "======================================================="
