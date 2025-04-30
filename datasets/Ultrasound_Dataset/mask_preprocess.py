import os
import re
import numpy as np
from PIL import Image
from collections import defaultdict
import shutil

# Define paths
mask_folder = 'datasets/Ultrasound_Dataset/Mask'
output_folder = 'datasets/Ultrasound_Dataset/Merged_Mask'
os.makedirs(output_folder, exist_ok=True)

# Regex patterns for base and additional parts
base_pattern = re.compile(r"(benign|malignant|normal) \((\d+)\)_mask\.png")
part_pattern = re.compile(r"(benign|malignant|normal) \((\d+)\)_mask_(\d+)\.png")

# Track groups for merging
mask_groups = defaultdict(list)
all_files = set(os.listdir(mask_folder))  # To track unmatched files

# Group masks
for fname in all_files:
    base = base_pattern.match(fname)
    part = part_pattern.match(fname)

    if base:
        category, idx = base.groups()
        mask_groups[(category, idx)].append(fname)
    elif part:
        category, idx, _ = part.groups()
        mask_groups[(category, idx)].append(fname)

# Process groups (merge or copy)
processed_files = set()

for (category, idx), files in mask_groups.items():
    merged = None
    has_multiple = any("_mask_" in f for f in files)

    for fname in files:
        processed_files.add(fname)
        path = os.path.join(mask_folder, fname)
        img = Image.open(path).convert("1")
        arr = np.array(img, dtype=np.uint8)

        if merged is None:
            merged = arr
        else:
            merged = np.logical_or(merged, arr).astype(np.uint8)

    output_name = f"{category} ({idx})_mask.png"
    output_path = os.path.join(output_folder, output_name)

    if has_multiple:
        # Save merged image
        merged_img = Image.fromarray((merged * 255).astype(np.uint8))
        merged_img.save(output_path)
    else:
        # Just copy the original file
        src = os.path.join(mask_folder, files[0])
        shutil.copy(src, output_path)

# Copy unmatched files (not part of any group)
for fname in all_files - processed_files:
    src = os.path.join(mask_folder, fname)
    dst = os.path.join(output_folder, fname)
    shutil.copy(src, dst)

print("All masks processed: merged and copied as needed.")
