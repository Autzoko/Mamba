#!/usr/bin/env python3
"""
Step 6: Failure case visualization.

For the 10 worst-case lesions (lowest IoU), generate a multi-panel figure:
  - Top row: axial / coronal / sagittal full-volume slices at GT bbox center
    with GT bbox (green) and pred bbox (orange dashed)
  - Bottom row: zoomed-in crop around GT bbox region
  - Title: volume ID, lesion ID, IoU score, GT bbox size

Usage:
    python visualize_failures.py \
        --results_csv /path/to/results/per_lesion_results.csv \
        --duying_root /path/to/Duying \
        --pred_dir /path/to/predictions_original_spacing \
        --output_dir /path/to/visualization \
        --n_worst 10
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
import pandas as pd


def find_volume(case_name: str, duying_root: Path) -> Path:
    """Find original Duying volume."""
    fname = f"{case_name}_0000.nii.gz"
    for subdir in ["raw_splitted/imagesTr", "raw_splitted/imagesTs"]:
        path = duying_root / subdir / fname
        if path.exists():
            return path
    raise FileNotFoundError(f"Volume not found: {case_name}")


def add_bbox_rect(ax, bbox, color, linestyle='-', linewidth=2):
    """Add a rectangle patch to an axis. bbox = (min_coord, max_coord) for the 2D slice."""
    y_min, x_min = bbox[0]
    y_max, x_max = bbox[1]
    width = x_max - x_min
    height = y_max - y_min
    rect = patches.Rectangle(
        (x_min, y_min), width, height,
        linewidth=linewidth, edgecolor=color, facecolor='none', linestyle=linestyle
    )
    ax.add_patch(rect)


def visualize_lesion(row, duying_root, pred_dir, output_dir, rank):
    """Generate a 2x3 figure for one lesion."""
    case_name = row["case_id"]

    # Load volume
    vol_path = find_volume(case_name, duying_root)
    vol_data = nib.load(str(vol_path)).get_fdata()

    # GT bbox
    gt = {
        "x1": int(row["gt_x1"]), "y1": int(row["gt_y1"]), "z1": int(row["gt_z1"]),
        "x2": int(row["gt_x2"]), "y2": int(row["gt_y2"]), "z2": int(row["gt_z2"]),
    }

    # Pred bbox
    has_pred = row["matched"]
    pred = None
    if has_pred:
        pred = {
            "x1": int(row["pred_x1"]), "y1": int(row["pred_y1"]), "z1": int(row["pred_z1"]),
            "x2": int(row["pred_x2"]), "y2": int(row["pred_y2"]), "z2": int(row["pred_z2"]),
        }

    # GT bbox center
    cx = (gt["x1"] + gt["x2"]) // 2
    cy = (gt["y1"] + gt["y2"]) // 2
    cz = (gt["z1"] + gt["z2"]) // 2

    # Clamp to volume bounds
    cx = np.clip(cx, 0, vol_data.shape[0] - 1)
    cy = np.clip(cy, 0, vol_data.shape[1] - 1)
    cz = np.clip(cz, 0, vol_data.shape[2] - 1)

    # Extract slices at GT center
    # Axial: fixed x -> vol[cx, :, :]
    # Coronal: fixed y -> vol[:, cy, :]
    # Sagittal: fixed z -> vol[:, :, cz]
    axial_slice = vol_data[cx, :, :]
    coronal_slice = vol_data[:, cy, :]
    sagittal_slice = vol_data[:, :, cz]

    # Normalize for display
    vmin, vmax = np.percentile(vol_data[vol_data > 0], [1, 99]) if (vol_data > 0).any() else (0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    gt_size = (gt["x2"] - gt["x1"] + 1, gt["y2"] - gt["y1"] + 1, gt["z2"] - gt["z1"] + 1)
    iou = row["iou"]
    inst_id = int(row["instance_id"])

    fig.suptitle(
        f"Rank #{rank} | {case_name} | Lesion {inst_id} | "
        f"IoU={iou:.4f} | GT size: {gt_size[0]}x{gt_size[1]}x{gt_size[2]}",
        fontsize=14, fontweight='bold'
    )

    views = [
        ("Axial (X={})".format(cx), axial_slice,
         (gt["y1"], gt["z1"]), (gt["y2"], gt["z2"]),
         (pred["y1"], pred["z1"]) if pred else None,
         (pred["y2"], pred["z2"]) if pred else None),
        ("Coronal (Y={})".format(cy), coronal_slice,
         (gt["x1"], gt["z1"]), (gt["x2"], gt["z2"]),
         (pred["x1"], pred["z1"]) if pred else None,
         (pred["x2"], pred["z2"]) if pred else None),
        ("Sagittal (Z={})".format(cz), sagittal_slice,
         (gt["x1"], gt["y1"]), (gt["x2"], gt["y2"]),
         (pred["x1"], pred["y1"]) if pred else None,
         (pred["x2"], pred["y2"]) if pred else None),
    ]

    pad = 30  # padding for zoomed view

    for col, (title, slc, gt_min, gt_max, pred_min, pred_max) in enumerate(views):
        # Top row: full view
        ax = axes[0, col]
        ax.imshow(slc.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        add_bbox_rect(ax, (gt_min, gt_max), color='lime', linestyle='-', linewidth=2)
        if pred_min is not None:
            add_bbox_rect(ax, (pred_min, pred_max), color='orange', linestyle='--', linewidth=2)
        ax.set_title(f"{title} (full)", fontsize=11)
        ax.axis('off')

        # Bottom row: zoomed crop
        ax = axes[1, col]
        # Compute zoom region
        all_mins = list(gt_min)
        all_maxs = list(gt_max)
        if pred_min is not None:
            all_mins = [min(a, b) for a, b in zip(all_mins, pred_min)]
            all_maxs = [max(a, b) for a, b in zip(all_maxs, pred_max)]

        r_min = max(0, all_mins[0] - pad)
        c_min = max(0, all_mins[1] - pad)
        r_max = min(slc.shape[0], all_maxs[0] + pad)
        c_max = min(slc.shape[1], all_maxs[1] + pad)

        cropped = slc[r_min:r_max, c_min:c_max]
        ax.imshow(cropped.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')

        # Adjust bbox coordinates for crop
        gt_min_adj = (gt_min[0] - r_min, gt_min[1] - c_min)
        gt_max_adj = (gt_max[0] - r_min, gt_max[1] - c_min)
        add_bbox_rect(ax, (gt_min_adj, gt_max_adj), color='lime', linestyle='-', linewidth=2)

        if pred_min is not None:
            pred_min_adj = (pred_min[0] - r_min, pred_min[1] - c_min)
            pred_max_adj = (pred_max[0] - r_min, pred_max[1] - c_min)
            add_bbox_rect(ax, (pred_min_adj, pred_max_adj), color='orange', linestyle='--', linewidth=2)

        ax.set_title(f"{title} (zoomed)", fontsize=11)
        ax.axis('off')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lime', linewidth=2, label='GT bbox'),
        Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='Pred bbox'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = output_dir / f"worst_{rank:02d}_{case_name}_inst{inst_id}_iou{iou:.3f}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Visualize worst failure cases")
    parser.add_argument("--results_csv", type=str, required=True,
                        help="Path to per_lesion_results.csv")
    parser.add_argument("--duying_root", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Dir with predicted masks at original Duying spacing")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_worst", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)

    # Sort by IoU ascending -> worst cases first
    df_sorted = df.sort_values("iou", ascending=True).head(args.n_worst)

    print(f"Generating {args.n_worst} worst-case visualizations...")

    for rank, (_, row) in enumerate(df_sorted.iterrows(), start=1):
        print(f"\n[{rank}/{args.n_worst}] {row['case_id']} inst={int(row['instance_id'])} "
              f"IoU={row['iou']:.4f}")
        try:
            visualize_lesion(row, Path(args.duying_root), Path(args.pred_dir), output_dir, rank)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
