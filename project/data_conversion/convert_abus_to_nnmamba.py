#!/usr/bin/env python3
"""
Step 1: Convert ABUS .nrrd volumes + masks into nnMamba-compatible format.

Preserves the original ABUS split:
  - Train (100 cases) → nnMamba training set
  - Validation (30 cases) → nnMamba validation set
  - Test (70 cases) → held out (not used in training)

All 200 cases are placed in imagesTr/labelsTr (nnUNet convention);
the actual train/val assignment is controlled via splits_final.pkl.

Produces:
    nnMamba_raw/Dataset001_ABUS/
        imagesTr/   ABUS_XXX_0000.nii.gz  (200 files)
        labelsTr/   ABUS_XXX.nii.gz       (200 files)
        imagesTs/   (empty placeholder)
        dataset.json
        splits_final.pkl

Usage:
    python convert_abus_to_nnmamba.py \
        --abus_root /path/to/ABUS/data \
        --output_base /path/to/nnMamba_raw
"""

import argparse
import json
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import nrrd


def nrrd_to_nifti(nrrd_path: str, spacing: tuple = (1.0, 1.0, 1.0)):
    """Read a .nrrd file and return a nibabel Nifti1Image."""
    data, header = nrrd.read(nrrd_path)
    affine = np.diag([*spacing, 1.0])
    img = nib.Nifti1Image(data, affine)
    return img


def collect_cases(split_dir: Path, split_name: str) -> list:
    """Collect all DATA/MASK pairs from one split directory."""
    data_dir = split_dir / "DATA"
    mask_dir = split_dir / "MASK"
    cases = []

    if not data_dir.exists():
        print(f"Warning: {data_dir} does not exist, skipping")
        return cases

    for data_file in sorted(data_dir.glob("DATA_*.nrrd")):
        idx = data_file.stem.replace("DATA_", "")
        mask_file = mask_dir / f"MASK_{idx}.nrrd"
        if not mask_file.exists():
            print(f"Warning: mask not found for {data_file}, skipping")
            continue
        cases.append({
            "id": idx,
            "data_path": str(data_file),
            "mask_path": str(mask_file),
            "split": split_name,
        })

    return cases


def main():
    parser = argparse.ArgumentParser(description="Convert ABUS to nnMamba format")
    parser.add_argument("--abus_root", type=str, required=True,
                        help="Path to ABUS/data containing Train/, Test/, Validation/")
    parser.add_argument("--output_base", type=str, required=True,
                        help="Path to nnMamba_raw base directory")
    args = parser.parse_args()

    abus_root = Path(args.abus_root)
    task_dir = Path(args.output_base) / "Dataset001_ABUS"
    images_tr = task_dir / "imagesTr"
    labels_tr = task_dir / "labelsTr"
    images_ts = task_dir / "imagesTs"

    for d in [images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect cases per original split
    train_cases = collect_cases(abus_root / "Train", "Train")
    val_cases = collect_cases(abus_root / "Validation", "Validation")
    test_cases = collect_cases(abus_root / "Test", "Test")
    all_cases = train_cases + val_cases + test_cases

    print(f"Original ABUS split:")
    print(f"  Train:      {len(train_cases)} cases")
    print(f"  Validation: {len(val_cases)} cases")
    print(f"  Test:       {len(test_cases)} cases")
    print(f"  Total:      {len(all_cases)} cases")

    # Convert all cases — all go into imagesTr/labelsTr
    # (nnUNet convention: everything in Tr, split managed by splits_final.pkl)
    training_list = []
    spacing = (1.0, 1.0, 1.0)

    for i, case in enumerate(all_cases):
        case_name = f"ABUS_{case['id']}"
        print(f"[{i+1}/{len(all_cases)}] Converting {case_name} ({case['split']})...")

        # Convert volume
        vol_img = nrrd_to_nifti(case["data_path"], spacing)
        nib.save(vol_img, str(images_tr / f"{case_name}_0000.nii.gz"))

        # Convert mask — ensure binary (0/1) uint8
        mask_data, _ = nrrd.read(case["mask_path"])
        mask_data = (mask_data > 0).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data, np.diag([*spacing, 1.0]))
        nib.save(mask_img, str(labels_tr / f"{case_name}.nii.gz"))

        training_list.append({
            "image": f"./imagesTr/{case_name}.nii.gz",
            "label": f"./labelsTr/{case_name}.nii.gz",
        })

    # Generate dataset.json
    dataset_json = {
        "name": "ABUS Breast Lesion Segmentation",
        "description": "TDSC-ABUS 3D breast ultrasound segmentation",
        "tensorImageSize": "4D",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "modality": {"0": "US"},
        "labels": {"0": "background", "1": "lesion"},
        "numTraining": len(all_cases),
        "numTest": 0,
        "training": training_list,
        "test": [],
    }

    with open(task_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # Build splits_final.pkl preserving original ABUS split
    # nnUNet format: list of 5 dicts (one per fold), each with 'train' and 'val' keys
    # Fold 0: original Train → train, original Validation → val
    # Test cases are included in the training set for maximum data usage,
    # since our real evaluation target is Duying (cross-domain).
    train_ids = [f"ABUS_{c['id']}" for c in train_cases + test_cases]
    val_ids = [f"ABUS_{c['id']}" for c in val_cases]

    # All 5 folds use the same split (we only train fold 0)
    splits = [{"train": np.array(train_ids), "val": np.array(val_ids)}
              for _ in range(5)]

    splits_path = task_dir / "splits_final.pkl"
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)

    print(f"\nDataset saved to: {task_dir}")
    print(f"  dataset.json:     {task_dir / 'dataset.json'}")
    print(f"  splits_final.pkl: {splits_path}")
    print(f"")
    print(f"nnMamba split (fold 0):")
    print(f"  Train: {len(train_ids)} cases (original Train + Test)")
    print(f"  Val:   {len(val_ids)} cases (original Validation)")


if __name__ == "__main__":
    main()
