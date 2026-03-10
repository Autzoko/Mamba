#!/usr/bin/env python3
"""
Step 4b: Resample predicted masks from isotropic 1x1x1mm back to
original Duying spacing (read from original volume headers).

Usage:
    python postprocess_predictions.py \
        --pred_dir /path/to/predictions_isotropic \
        --duying_root /path/to/Duying \
        --output_dir /path/to/predictions_original_spacing
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def find_original_volume(case_name: str, duying_root: Path) -> Path:
    """Find the original Duying volume for a given case name."""
    fname = f"{case_name}_0000.nii.gz"
    for subdir in ["raw_splitted/imagesTr", "raw_splitted/imagesTs"]:
        path = duying_root / subdir / fname
        if path.exists():
            return path
    raise FileNotFoundError(f"Original volume not found for {case_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--duying_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    duying_root = Path(args.duying_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("case_*.nii.gz"))
    # Exclude softmax files if any
    pred_files = [f for f in pred_files if "softmax" not in f.name]

    print(f"Found {len(pred_files)} prediction files")

    for i, pred_path in enumerate(pred_files):
        case_name = pred_path.name.replace(".nii.gz", "")
        print(f"[{i+1}/{len(pred_files)}] Resampling {case_name}...")

        try:
            # Load prediction (isotropic 1x1x1mm)
            pred_img = nib.load(str(pred_path))
            pred_data = pred_img.get_fdata().astype(np.uint8)
            pred_spacing = tuple(pred_img.header.get_zooms()[:3])

            # Load original volume to get target shape and spacing
            orig_path = find_original_volume(case_name, duying_root)
            orig_img = nib.load(str(orig_path))
            orig_shape = orig_img.shape[:3]
            orig_spacing = tuple(orig_img.header.get_zooms()[:3])

            # Resample prediction to original shape
            # Use nearest neighbor for segmentation masks
            zoom_factors = [pred_spacing[i] / orig_spacing[i]
                            for i in range(3)]
            # Actually we need to match original shape exactly
            zoom_factors = [orig_shape[i] / pred_data.shape[i]
                            for i in range(3)]

            resampled = zoom(pred_data.astype(np.float32), zoom_factors,
                             order=0, mode='nearest')
            resampled = (resampled > 0.5).astype(np.uint8)

            # Ensure shape matches exactly (zoom can be off by 1)
            if resampled.shape != orig_shape:
                result = np.zeros(orig_shape, dtype=np.uint8)
                slices = tuple(slice(0, min(r, o))
                               for r, o in zip(resampled.shape, orig_shape))
                result[slices] = resampled[slices]
                resampled = result

            # Save with original affine
            out_img = nib.Nifti1Image(resampled, orig_img.affine, orig_img.header)
            nib.save(out_img, str(output_dir / pred_path.name))

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Resampled predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
