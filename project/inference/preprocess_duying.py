#!/usr/bin/env python3
"""
Step 3: Preprocess Duying volumes for nnMamba inference.

For each Duying volume:
  1. Resample from 1x3x1mm to isotropic 1x1x1mm (trilinear interpolation)
  2. Apply the same intensity normalization as ABUS training
     (percentile-based clip at 0.5th and 99.5th percentile, then z-score)
  3. Save as case_XXXXX_0000.nii.gz in output directory

Usage:
    python preprocess_duying.py \
        --duying_root /path/to/Duying \
        --output_dir /path/to/duying_preprocessed \
        --num_workers 8
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def resample_volume(data: np.ndarray, original_spacing: tuple, target_spacing: tuple,
                    order: int = 1) -> np.ndarray:
    """Resample volume from original_spacing to target_spacing."""
    zoom_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
    resampled = zoom(data, zoom_factors, order=order, mode='nearest')
    return resampled


def normalize_intensity(data: np.ndarray, lower_pct: float = 0.5, upper_pct: float = 99.5) -> np.ndarray:
    """
    Percentile-based intensity normalization (same as nnUNet non-CT scheme).
    Clip at lower/upper percentile, then z-score normalize using the clipped data.
    """
    # Compute percentiles on foreground (nonzero) voxels if available
    mask = data > 0
    if mask.sum() > 100:
        foreground = data[mask]
    else:
        foreground = data.flatten()

    lower = np.percentile(foreground, lower_pct)
    upper = np.percentile(foreground, upper_pct)

    data = np.clip(data, lower, upper)

    mean_val = data[mask].mean() if mask.sum() > 100 else data.mean()
    std_val = data[mask].std() if mask.sum() > 100 else data.std()

    if std_val < 1e-8:
        std_val = 1.0

    data = (data - mean_val) / std_val
    return data


def process_single_volume(args_tuple):
    """Process a single Duying volume: resample + normalize + save."""
    nii_path, output_dir, target_spacing = args_tuple
    fname = Path(nii_path).name  # case_XXXXX_0000.nii.gz

    try:
        img = nib.load(str(nii_path))
        data = img.get_fdata().astype(np.float32)

        # Get original spacing from header
        header = img.header
        original_spacing = tuple(header.get_zooms()[:3])

        # Resample to target spacing
        if not all(abs(o - t) < 0.01 for o, t in zip(original_spacing, target_spacing)):
            data = resample_volume(data, original_spacing, target_spacing, order=1)

        # Normalize intensity
        data = normalize_intensity(data)

        # Save with target spacing
        affine = np.diag([*target_spacing, 1.0])
        out_img = nib.Nifti1Image(data, affine)
        out_path = Path(output_dir) / fname
        nib.save(out_img, str(out_path))

        return fname, original_spacing, data.shape, "OK"
    except Exception as e:
        return fname, None, None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Preprocess Duying volumes for nnMamba inference")
    parser.add_argument("--duying_root", type=str, required=True,
                        help="Path to Duying dataset root")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for preprocessed volumes")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Target isotropic spacing (default: 1 1 1)")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_spacing = tuple(args.target_spacing)

    # Collect all Duying volumes from both imagesTr and imagesTs
    duying_root = Path(args.duying_root)
    volume_paths = []
    for subdir in ["raw_splitted/imagesTr", "raw_splitted/imagesTs"]:
        vol_dir = duying_root / subdir
        if vol_dir.exists():
            volume_paths.extend(sorted(vol_dir.glob("case_*_0000.nii.gz")))

    print(f"Found {len(volume_paths)} Duying volumes")
    print(f"Target spacing: {target_spacing}")
    print(f"Output: {output_dir}")

    # Process volumes in parallel
    tasks = [(str(p), str(output_dir), target_spacing) for p in volume_paths]

    completed = 0
    errors = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_volume, t): t for t in tasks}
        for future in as_completed(futures):
            fname, orig_sp, shape, status = future.result()
            completed += 1
            if status == "OK":
                print(f"[{completed}/{len(tasks)}] {fname}: "
                      f"spacing {orig_sp} -> {target_spacing}, shape {shape}")
            else:
                print(f"[{completed}/{len(tasks)}] {fname}: ERROR - {status}")
                errors.append((fname, status))

    print(f"\nDone. Processed {completed - len(errors)}/{len(tasks)} successfully.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for fname, err in errors:
            print(f"  {fname}: {err}")


if __name__ == "__main__":
    main()
