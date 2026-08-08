# Reliability-Guided Depth Estimation for Surgical Tool Tracking in Endoscopic Images

This project investigates whether segmentation information can be used to improve monocular depth estimation in endoscopic scenes. The proposed framework uses SAM-based surgical tool segmentation to estimate unreliable image regions and applies a reliability-based weighting scheme before depth inference. It is built on [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

---

## ✨ Overview

This project implements a reliability-guided depth estimation pipeline for endoscopic surgical scenes. The pipeline estimates a pixel-wise reliability map from image appearance cues including shading, texture, edges, and specular highlights. The estimated reliability map is used to modulate the input image before depth inference. Raw and reliability-guided depth predictions are then compared using quantitative metrics.

---

## 🗂 Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

## 🚀 Installation

```bash
# 1. Clone and enter the repository
git clone https://github.com/JZchn/Depth_estimation_conditioned_by_Segmentation_Project.git
cd Depth_estimation_conditioned_by_Segmentation_Project

# 2. Download model weights
bash download_weights.sh

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Clone Depth Anything V2 (third-party dependency)
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

## 📁 Running the Code

| Script | Purpose | Command |
|--------|---------|---------|
| `run_demo.py` | Demo with synthetic data (no real images required) | `python run_demo.py` |
| `run_local_eval.py` | Full evaluation on real endoscopic data (50 images with GUI point selection) | `python run_local_eval.py` |
| `run_eval_metrics.py` | Batch statistics (RMSE/MAE/AbsRel/wRMSE, paired t-test over 50 images) | `python run_eval_metrics.py` |

---

## 🗂 Project Structure

```
├── run_demo.py               # Demo script (synthetic data)
├── run_local_eval.py         # Evaluation script (real data)
├── run_eval_metrics.py       # Batch metric computation & statistics
├── src/
│   ├── config.py             # Configuration & paths
│   ├── sam_segmentation.py   # SAM-based tool segmentation
│   ├── depth_inference.py    # Depth Anything V2 wrapper
│   ├── reliability.py        # Reliability feature extraction
│   ├── metrics.py            # Evaluation metrics & batch statistics
│   ├── tilt_analysis.py      # Tool tilt angle estimation
│   └── visualization.py      # Publication-quality figure generation
├── requirements.txt
├── download_weights.sh
├── LICENSE
└── README.md
```

---

## 📊 Data Privacy Notice

**This repository does not include any medical images.** The `data/` directory is listed in `.gitignore` and must not be uploaded. The demo script (`run_demo.py`) generates purely synthetic images for illustration. Real endoscopic data required for `run_local_eval.py` must be obtained separately from authorized sources.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — depth estimation backbone
- [Segment Anything Model (SAM)](https://segment-anything.com/) — tool segmentation
