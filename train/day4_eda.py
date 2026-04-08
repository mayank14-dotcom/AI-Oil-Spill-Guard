# DAY 3 WAS NOTHNG THIS IS DAY4 Module 2: Data Exploration and Data Preprocessing
import os

# actual location: images and masks are in the same train directory
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = script_dir

# filter for only image files (not masks or other files)
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
image_files = sorted([f for f in os.listdir(train_path)
                      if os.path.isfile(os.path.join(train_path, f))
                      and os.path.splitext(f)[1].lower() in image_extensions
                      and not f.endswith('_mask.png')])

mask_files = sorted([f for f in os.listdir(train_path)
                     if os.path.isfile(os.path.join(train_path, f))
                     and f.endswith('_mask.png')])

print("Total Images:", len(image_files))
print("Total Masks:", len(mask_files))

# Check if each image has a corresponding mask
missing_masks = []
matched_pairs = 0

for img in image_files:
    base_name, ext = os.path.splitext(img)
    expected_mask = base_name + "_mask.png"
    
    if expected_mask in mask_files:
        matched_pairs += 1
    else:
        missing_masks.append((img, f"Missing: {expected_mask}"))

print(f"\n Matched image-mask pairs: {matched_pairs}")

if len(missing_masks) == 0:
    print("All images have corresponding masks!")
else:
    print(f" Found {len(missing_masks)} images without masks:")
    for img, msg in missing_masks[:5]:
        print(f"  - {img} -> {msg}")

# display size information for the first few pairs
from PIL import Image
print("\nChecking first 5 image/mask sizes:")
for file in image_files[:5]:
    img = Image.open(os.path.join(train_path, file))
    mask_name = os.path.splitext(file)[0] + '_mask.png'
    mask = Image.open(os.path.join(train_path, mask_name))

    print(file)
    print("Image Size:", img.size)
    print("Mask Size :", mask.size)
    print("Match:", img.size == mask.size)
    print("--------------")

# additional analysis: binary mask and visualization
import numpy as np
import matplotlib.pyplot as plt
import random

# helper to convert mask to 0/255 binary

def convert_to_binary(mask):
    mask = np.array(mask)
    mask = np.where(mask > 127, 255, 0)  # thresholding
    return mask

# randomly sample a few images for display
sample_files = random.sample(image_files, 5)

for file in sample_files:
    img = Image.open(os.path.join(train_path, file))
    mask_name = os.path.splitext(file)[0] + '_mask.png'
    mask = Image.open(os.path.join(train_path, mask_name))
    mask = convert_to_binary(mask)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Image")

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap="gray")
    plt.title("Binary Mask")

    plt.show()

# overlay view
for file in sample_files:
    img = Image.open(os.path.join(train_path, file))
    mask_name = os.path.splitext(file)[0] + '_mask.png'
    mask = Image.open(os.path.join(train_path, mask_name))
    mask = convert_to_binary(mask)

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.imshow(mask, alpha=0.4, cmap="Reds")
    plt.title("Overlay (Oil highlighted)")
    plt.show()

# compute overall pixel statistics
oil_pixels = 0
total_pixels = 0

for file in image_files:
    mask_name = os.path.splitext(file)[0] + '_mask.png'
    mask = Image.open(os.path.join(train_path, mask_name))
    mask = convert_to_binary(mask)

    oil_pixels += np.sum(mask == 255)
    total_pixels += mask.size

background_pixels = total_pixels - oil_pixels

oil_percent = (oil_pixels / total_pixels) * 100
bg_percent = 100 - oil_percent

print("Total Pixels:", total_pixels)
print("Oil Pixels:", oil_pixels)
print("Background Pixels:", background_pixels)

print(f"Oil Percentage: {oil_percent:.2f}%")
print(f"Background Percentage: {bg_percent:.2f}%")

plt.bar(["Oil", "Background"], [oil_percent, bg_percent])
plt.title("Pixel Distribution (%)")
plt.ylabel("Percentage")
plt.show()