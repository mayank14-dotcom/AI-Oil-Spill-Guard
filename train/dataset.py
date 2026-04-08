# DAY2 Module 2: Data Exploration and Data Preprocessing

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode, PILToTensor

def _infer_resize_size(transform):
    """Try to extract the resize target size from a torchvision transform."""
    if transform is None:
        return None
    if isinstance(transform, transforms.Resize):
        return transform.size
    if isinstance(transform, transforms.Compose):
        for t in transform.transforms:
            if isinstance(t, transforms.Resize):
                return t.size
    return None

class OilDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # list only actual image files (exclude masks and other files)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        self.images = sorted([f for f in os.listdir(image_dir)
                              if os.path.isfile(os.path.join(image_dir, f))
                              and os.path.splitext(f)[1].lower() in image_extensions
                              and not f.endswith('_mask.png')])  # exclude mask files
        self.transform = transform
        # build a safe default mask transform if none is provided
        if mask_transform is None:
            resize_size = _infer_resize_size(transform)
            if resize_size is not None:
                mask_transform = transforms.Compose([
                    transforms.Resize(resize_size, interpolation=InterpolationMode.NEAREST),
                    PILToTensor(),
                ])
            else:
                mask_transform = transforms.Compose([
                    PILToTensor(),
                ])
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)

        # masks in this dataset are named like <basename>_mask.png
        base_name, _ = os.path.splitext(img_name)
        mask_name = base_name + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Add error handling for missing/corrupted files
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path} or mask {mask_path}: {e}")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # ensure mask is tensor and binary
        if not isinstance(mask, torch.Tensor):
            mask = PILToTensor()(mask)
        # if mask is multi-channel, keep the first channel
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask = mask[:1]
        mask = (mask > 0).float()

        return image, mask


if __name__ == "__main__":
    # simple transformation for both images and masks (resize + tensor)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # 0-1 scaling
    ])

    # paths can be changed as needed; make them absolute or relative to script
    # in this archive the images and masks live directly under 'train'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(script_dir)  # this file is already in train/
    img_dir = base   # use the train folder as the image directory
    mask_dir = base  # masks are intermixed; OilDataset will append '_mask.png'

    # verify directories exist
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"image directory not found: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"mask directory not found: {mask_dir}")

    dataset = OilDataset(img_dir, mask_dir, transform=transform)

    # split into train/val/test
    from torch.utils.data import random_split

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))

    # compute oil / background percentage across entire set
    oil_pixels = 0
    total_pixels = 0

    for img, mask in dataset:
        oil_pixels += torch.sum(mask)
        total_pixels += torch.numel(mask)

    oil_percent = (oil_pixels / total_pixels) * 100
    bg_percent = 100 - oil_percent

    print(f"Oil %: {oil_percent:.2f}")
    print(f"Background %: {bg_percent:.2f}")
