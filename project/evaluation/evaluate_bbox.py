#!/usr/bin/env python3
"""
Step 5: Evaluate predicted masks vs GT bounding boxes.

For each predicted mask:
  - Extract 3D tight bounding boxes of connected components
  - Load Duying GT bounding boxes from CSV
  - Match predicted bboxes to GT bboxes by 3D IoU
  - Compute per-lesion IoU, recall, precision
  - Summarize statistics and detection rates

Usage:
    python evaluate_bbox.py \
        --pred_dir /path/to/predictions_original_spacing \
        --duying_root /path/to/Duying \
        --output_dir /path/to/results
"""

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_components


def extract_bboxes_from_mask(mask: np.ndarray, min_volume: int = 10):
    """
    Extract tight 3D bounding boxes from connected components in a binary mask.
    Returns list of dicts with keys: x1, y1, z1, x2, y2, z2, volume.
    """
    labeled, num_features = connected_components(mask > 0)
    bboxes = []
    for comp_id in range(1, num_features + 1):
        coords = np.argwhere(labeled == comp_id)
        vol = len(coords)
        if vol < min_volume:
            continue
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        bboxes.append({
            "x1": int(mins[0]), "y1": int(mins[1]), "z1": int(mins[2]),
            "x2": int(maxs[0]), "y2": int(maxs[1]), "z2": int(maxs[2]),
            "volume": vol,
        })
    return bboxes


def bbox_volume(b):
    """Volume of a bounding box (in voxels)."""
    return max(0, b["x2"] - b["x1"] + 1) * \
           max(0, b["y2"] - b["y1"] + 1) * \
           max(0, b["z2"] - b["z1"] + 1)


def bbox_intersection(a, b):
    """Intersection volume of two bounding boxes."""
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    z1 = max(a["z1"], b["z1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    z2 = min(a["z2"], b["z2"])

    if x1 > x2 or y1 > y2 or z1 > z2:
        return 0
    return (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)


def bbox_iou(a, b):
    """3D IoU between two bounding boxes."""
    inter = bbox_intersection(a, b)
    union = bbox_volume(a) + bbox_volume(b) - inter
    if union == 0:
        return 0.0
    return inter / union


def bbox_recall(gt, pred):
    """Fraction of GT bbox volume covered by pred bbox."""
    inter = bbox_intersection(gt, pred)
    gt_vol = bbox_volume(gt)
    if gt_vol == 0:
        return 0.0
    return inter / gt_vol


def bbox_precision(gt, pred):
    """Fraction of pred bbox volume that falls within GT bbox."""
    inter = bbox_intersection(gt, pred)
    pred_vol = bbox_volume(pred)
    if pred_vol == 0:
        return 0.0
    return inter / pred_vol


def load_duying_gt_bboxes(duying_root: Path) -> dict:
    """
    Load all GT bounding boxes from Duying CSV files.
    Returns: dict[case_id] -> list of bbox dicts
    Coordinates in CSV are in voxel space (x1, y1, z1, x2, y2, z2).
    """
    gt = {}
    for csv_name in ["bboxes_train.csv", "bboxes_test.csv"]:
        csv_path = duying_root / csv_name
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        for _, row in df.iterrows():
            case_id = str(row["case_id"]).strip()
            bbox = {
                "x1": int(row["x1"]), "y1": int(row["y1"]), "z1": int(row["z1"]),
                "x2": int(row["x2"]), "y2": int(row["y2"]), "z2": int(row["z2"]),
                "instance_id": int(row["instance_id"]),
            }
            if case_id not in gt:
                gt[case_id] = []
            gt[case_id].append(bbox)

    return gt


def main():
    parser = argparse.ArgumentParser(description="Evaluate pred masks vs GT bboxes")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Dir with predicted masks at original Duying spacing")
    parser.add_argument("--duying_root", type=str, required=True,
                        help="Duying dataset root")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--min_component_volume", type=int, default=10,
                        help="Min voxels for a connected component to be considered")
    parser.add_argument("--largest_only", action="store_true",
                        help="Keep only the largest connected component per case")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    duying_root = Path(args.duying_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GT bounding boxes
    gt_bboxes = load_duying_gt_bboxes(duying_root)
    total_gt_lesions = sum(len(v) for v in gt_bboxes.values())
    print(f"Loaded GT bboxes for {len(gt_bboxes)} cases, {total_gt_lesions} total lesions")

    # Process each prediction
    pred_files = sorted(pred_dir.glob("case_*.nii.gz"))
    print(f"Found {len(pred_files)} prediction files")

    all_results = []
    unmatched_gt = 0
    cases_without_gt = 0

    for pi, pred_path in enumerate(pred_files):
        case_name = pred_path.name.replace(".nii.gz", "")
        print(f"[{pi+1}/{len(pred_files)}] {case_name}...", flush=True)

        # Load prediction mask
        pred_img = nib.load(str(pred_path))
        pred_mask = pred_img.get_fdata().astype(np.uint8)

        # Extract predicted bboxes
        pred_bboxes = extract_bboxes_from_mask(pred_mask, args.min_component_volume)

        # Optionally keep only the largest component
        if args.largest_only and len(pred_bboxes) > 1:
            pred_bboxes = [max(pred_bboxes, key=lambda b: b["volume"])]

        # Get GT bboxes for this case
        if case_name not in gt_bboxes:
            cases_without_gt += 1
            continue

        gt_list = gt_bboxes[case_name]

        # For each GT bbox, find best matching predicted bbox by IoU
        for gt_bbox in gt_list:
            best_iou = 0.0
            best_recall = 0.0
            best_precision = 0.0
            best_pred_bbox = None

            for pred_bbox in pred_bboxes:
                iou = bbox_iou(gt_bbox, pred_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_recall = bbox_recall(gt_bbox, pred_bbox)
                    best_precision = bbox_precision(gt_bbox, pred_bbox)
                    best_pred_bbox = pred_bbox

            gt_size = (
                gt_bbox["x2"] - gt_bbox["x1"] + 1,
                gt_bbox["y2"] - gt_bbox["y1"] + 1,
                gt_bbox["z2"] - gt_bbox["z1"] + 1,
            )

            result = {
                "case_id": case_name,
                "instance_id": gt_bbox.get("instance_id", -1),
                "iou": best_iou,
                "recall": best_recall,
                "precision": best_precision,
                "gt_x1": gt_bbox["x1"], "gt_y1": gt_bbox["y1"], "gt_z1": gt_bbox["z1"],
                "gt_x2": gt_bbox["x2"], "gt_y2": gt_bbox["y2"], "gt_z2": gt_bbox["z2"],
                "gt_size_x": gt_size[0], "gt_size_y": gt_size[1], "gt_size_z": gt_size[2],
                "gt_volume": bbox_volume(gt_bbox),
                "num_pred_components": len(pred_bboxes),
                "matched": best_pred_bbox is not None and best_iou > 0,
            }

            if best_pred_bbox is not None:
                result.update({
                    "pred_x1": best_pred_bbox["x1"], "pred_y1": best_pred_bbox["y1"],
                    "pred_z1": best_pred_bbox["z1"],
                    "pred_x2": best_pred_bbox["x2"], "pred_y2": best_pred_bbox["y2"],
                    "pred_z2": best_pred_bbox["z2"],
                    "pred_volume": bbox_volume(best_pred_bbox),
                })
            else:
                result.update({
                    "pred_x1": 0, "pred_y1": 0, "pred_z1": 0,
                    "pred_x2": 0, "pred_y2": 0, "pred_z2": 0,
                    "pred_volume": 0,
                })
                unmatched_gt += 1

            all_results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    if len(df) == 0:
        print("No results to evaluate!")
        return

    # Save per-lesion results
    df.to_csv(output_dir / "per_lesion_results.csv", index=False)

    # Compute summary statistics
    iou_vals = df["iou"].values
    recall_vals = df["recall"].values
    precision_vals = df["precision"].values

    summary = {
        "num_cases_evaluated": len(df["case_id"].unique()),
        "num_gt_lesions": len(df),
        "num_unmatched_gt": unmatched_gt,
        "cases_without_gt": cases_without_gt,
        "iou_mean": float(np.mean(iou_vals)),
        "iou_std": float(np.std(iou_vals)),
        "iou_median": float(np.median(iou_vals)),
        "recall_mean": float(np.mean(recall_vals)),
        "recall_std": float(np.std(recall_vals)),
        "precision_mean": float(np.mean(precision_vals)),
        "precision_std": float(np.std(precision_vals)),
        "detection_rate_iou_0.1": float((iou_vals > 0.1).mean()),
        "detection_rate_iou_0.3": float((iou_vals > 0.3).mean()),
        "detection_rate_iou_0.5": float((iou_vals > 0.5).mean()),
    }

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Cases evaluated:    {summary['num_cases_evaluated']}")
    print(f"GT lesions:         {summary['num_gt_lesions']}")
    print(f"Unmatched GT:       {summary['num_unmatched_gt']}")
    print(f"")
    print(f"3D IoU:             {summary['iou_mean']:.4f} +/- {summary['iou_std']:.4f}")
    print(f"3D IoU (median):    {summary['iou_median']:.4f}")
    print(f"Recall:             {summary['recall_mean']:.4f} +/- {summary['recall_std']:.4f}")
    print(f"Precision:          {summary['precision_mean']:.4f} +/- {summary['precision_std']:.4f}")
    print(f"")
    print(f"Detection rate:")
    print(f"  IoU > 0.1:        {summary['detection_rate_iou_0.1']:.4f} "
          f"({int(sum(iou_vals > 0.1))}/{len(iou_vals)})")
    print(f"  IoU > 0.3:        {summary['detection_rate_iou_0.3']:.4f} "
          f"({int(sum(iou_vals > 0.3))}/{len(iou_vals)})")
    print(f"  IoU > 0.5:        {summary['detection_rate_iou_0.5']:.4f} "
          f"({int(sum(iou_vals > 0.5))}/{len(iou_vals)})")
    print("=" * 60)

    print(f"\nResults saved to: {output_dir}")
    print(f"  per_lesion_results.csv")
    print(f"  summary.json")

    # Generate visualization
    print("\nGenerating plots...")
    generate_plots(df, output_dir)
    print(f"  Plots saved to {output_dir}")


def generate_plots(df, output_dir):
    """Generate evaluation visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. IoU histogram
    ax = axes[0, 0]
    ax.hist(df["iou"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("3D IoU")
    ax.set_ylabel("Count")
    ax.set_title(f"IoU Distribution (mean={df['iou'].mean():.3f})")
    ax.axvline(df["iou"].mean(), color="red", linestyle="--", label=f"mean={df['iou'].mean():.3f}")
    ax.axvline(df["iou"].median(), color="orange", linestyle="--", label=f"median={df['iou'].median():.3f}")
    ax.legend()

    # 2. Recall histogram
    ax = axes[0, 1]
    ax.hist(df["recall"], bins=50, edgecolor="black", alpha=0.7, color="seagreen")
    ax.set_xlabel("Recall (GT coverage)")
    ax.set_ylabel("Count")
    ax.set_title(f"Recall Distribution (mean={df['recall'].mean():.3f})")

    # 3. Precision histogram
    ax = axes[0, 2]
    ax.hist(df["precision"], bins=50, edgecolor="black", alpha=0.7, color="coral")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Count")
    ax.set_title(f"Precision Distribution (mean={df['precision'].mean():.3f})")

    # 4. IoU vs GT volume (scatter)
    ax = axes[1, 0]
    ax.scatter(df["gt_volume"], df["iou"], alpha=0.4, s=15, c="steelblue")
    ax.set_xlabel("GT Lesion Volume (voxels)")
    ax.set_ylabel("3D IoU")
    ax.set_title("IoU vs GT Lesion Size")
    ax.set_xscale("log")

    # 5. Detection rate at different IoU thresholds
    ax = axes[1, 1]
    thresholds = np.arange(0, 1.01, 0.05)
    rates = [(df["iou"] > t).mean() for t in thresholds]
    ax.plot(thresholds, rates, "o-", color="steelblue", markersize=4)
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection Rate vs IoU Threshold")
    ax.grid(True, alpha=0.3)
    for t in [0.1, 0.3, 0.5]:
        r = (df["iou"] > t).mean()
        ax.axvline(t, color="gray", linestyle=":", alpha=0.5)
        ax.annotate(f"{r:.2f}", (t, r), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    # 6. Number of predicted components per case
    ax = axes[1, 2]
    n_comps = df.groupby("case_id")["num_pred_components"].first()
    ax.hist(n_comps, bins=50, edgecolor="black", alpha=0.7, color="mediumpurple")
    ax.set_xlabel("# Predicted Components")
    ax.set_ylabel("# Cases")
    ax.set_title(f"Pred Components/Case (median={int(n_comps.median())})")

    plt.suptitle("Cross-Domain Evaluation: nnMamba (ABUS) -> Duying", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_plots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Also save a per-case summary plot (sorted by mean IoU)
    case_iou = df.groupby("case_id")["iou"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(case_iou)), case_iou.values, color="steelblue", alpha=0.7)
    ax.set_xlabel("Cases (sorted by mean IoU)")
    ax.set_ylabel("Mean IoU")
    ax.set_title(f"Per-Case Mean IoU ({len(case_iou)} cases)")
    ax.axhline(case_iou.mean(), color="red", linestyle="--", label=f"overall mean={case_iou.mean():.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_case_iou.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
