# DAY1 Module 1: Data Collection

import os, glob
from PIL import Image
import matplotlib.pyplot as plt
import random

base = r"C:\Users\Mayank\Downloads\archive (4)\train"        # no “images” sub‑folder
image_files = glob.glob(os.path.join(base, "*.jpg"))         # or *.png, *.jpeg…
mask_files  = glob.glob(os.path.join(base, "*_mask.png"))    # adjust pattern as needed

# Count Images & Masks
print("Number of images:", len(image_files))
print("Number of masks:",  len(mask_files))

# inspect a sample image if available
if image_files:
    sample_img = image_files[0]
    img = Image.open(sample_img)
    print("Sample image file:", sample_img)
    width, height = img.size
    print(f"Image size: {width} x {height} (width x height)")
    print(f"Resolution: {width}×{height} pixels")

# show a few random image/mask pairs
if image_files:
    # helper to build the mask path from an image path
    def corresponding_mask(img_path):
        fname = os.path.basename(img_path)
        base_name = os.path.splitext(fname)[0]
        mask_name = base_name + "_mask.png"
        return os.path.join(base, mask_name)

    for img_path in random.sample(image_files, min(5, len(image_files))):
        mask_path = corresponding_mask(img_path)
        if not os.path.exists(mask_path):
            print(f"mask not found for {img_path}")
            continue

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Image")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")

        plt.show()




