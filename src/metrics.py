import numpy as np

def sample_depth_robust(depth_map, x, y, r=3):
    """Sample median depth in local neighborhood to resist noise."""
    H, W = depth_map.shape
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    patch = depth_map[y0:y1, x0:x1]
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return np.nan
    return np.median(patch)

def compute_depth_metrics(depth_pred, depth_gt, valid_mask, reliability=None):
    """Compute RMSE, MAE, AbsRel, and weighted RMSE."""
    pred = depth_pred[valid_mask]
    gt = depth_gt[valid_mask]
    eps = 1e-8
    diff = pred - gt
    abs_diff = np.abs(diff)

    metrics = {}
    metrics["RMSE"] = float(np.sqrt(np.mean(diff ** 2)))
    metrics["MAE"] = float(np.mean(abs_diff))
    metrics["AbsRel"] = float(np.mean(abs_diff / (gt + eps)))

    if reliability is not None:
        w = reliability[valid_mask]
        metrics["wRMSE"] = float(np.sqrt(np.sum(w * diff ** 2) / (np.sum(w) + eps)))
        metrics["wAbsRel"] = float(np.sum(w * abs_diff / (gt + eps)) / (np.sum(w) + eps))

    return metrics


def compute_batch_statistics(raw_metrics_list, guided_metrics_list):
    """
    Compute batch-level statistics from per-image metric lists.
    
    Args:
        raw_metrics_list: list of dicts, each with keys RMSE, MAE, AbsRel, (optionally wRMSE, wAbsRel)
        guided_metrics_list: same structure for guided depth
    
    Returns:
        stats: dict of dicts, e.g. stats["RMSE"] = {"raw_mean": ..., "raw_std": ..., 
               "guided_mean": ..., "guided_std": ..., "p_value": ...}
    """
    from scipy import stats as scipy_stats
    
    # Get all metric keys present in the first dict (intersection of both lists)
    keys = [k for k in raw_metrics_list[0].keys() if k in guided_metrics_list[0]]
    
    stats = {}
    for key in keys:
        raw_vals = np.array([m[key] for m in raw_metrics_list], dtype=np.float64)
        guided_vals = np.array([m[key] for m in guided_metrics_list], dtype=np.float64)
        
        raw_mean = float(np.mean(raw_vals))
        raw_std = float(np.std(raw_vals, ddof=1))
        guided_mean = float(np.mean(guided_vals))
        guided_std = float(np.std(guided_vals, ddof=1))
        
        # Paired t-test (two-tailed)
        t_stat, p_value = scipy_stats.ttest_rel(raw_vals, guided_vals)
        
        stats[key] = {
            "raw_mean": raw_mean,
            "raw_std": raw_std,
            "guided_mean": guided_mean,
            "guided_std": guided_std,
            "p_value": float(p_value),
            "t_stat": float(t_stat),
        }
    
    return stats


def print_metrics_table(stats):
    """
    Print batch statistics as a formatted LaTeX/academic table.
    """
    header = f"{'Metric':>10} | {'Raw Mean':>9} {'Raw Std':>8} | {'Guided Mean':>12} {'Guided Std':>11} | {'P-value':>8}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
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
        print(f"{key:>10} | {s['raw_mean']:>8.4f} {s['raw_std']:>7.4f} | {s['guided_mean']:>8.4f} {s['guided_std']:>7.4f} | {s['p_value']:>6.4f} {sig}")
    print(sep + "\n")
