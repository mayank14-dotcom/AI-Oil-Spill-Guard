# ==============================
# MODULE 4: TRAINING & EVALUATION
# ==============================

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import UNet
from dataset import OilDataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode, PILToTensor

# ------------------------------
# Device
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Transforms
# ------------------------------
# Image transforms (match UI/inference)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Mask transforms (use NEAREST to preserve labels)
mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    PILToTensor(),
])

# ------------------------------
# Dataset & Split
# ------------------------------
# the files live directly in the "train" folder, not in separate
# subdirectories. compute a base path so the script works regardless
# of the current working directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = script_dir  # same directory contains both images and masks

dataset = OilDataset(base_dir, base_dir, transform=transform, mask_transform=mask_transform)

# Optional: limit dataset size for faster experiments (set MAX_SAMPLES env var)
max_samples_env = os.getenv("MAX_SAMPLES", "").strip()
if max_samples_env:
    try:
        max_samples = int(max_samples_env)
        if 0 < max_samples < len(dataset):
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = torch.utils.data.Subset(dataset, indices[:max_samples])
            print(f"Using a subset of {len(dataset)} samples for training.")
    except ValueError:
        print(f"Ignoring invalid MAX_SAMPLES value: {max_samples_env}")

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

# print summary so user knows the script started correctly
print(f"Full dataset: {len(dataset)} samples")
print(f"Split -> train: {train_size}, val: {val_size}, test: {test_size}")

batch_size = int(os.getenv("BATCH_SIZE", "4"))
val_batch_size = int(os.getenv("VAL_BATCH_SIZE", "8"))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=0)

# ------------------------------
# Model
# ------------------------------
print(f"Device: {device}")
print("Creating UNet model...")
model = UNet(in_channels=3, out_channels=1).to(device)
print("Model created and moved to device successfully")

# ------------------------------
# Loss Functions
# ------------------------------

# Loss helpers: avoid creating BCE module at import time so we can move
# the model to GPU without hitting a device mismatch.

def dice_loss(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

def combined_loss(pred, target):
    bce = nn.BCELoss()
    if pred.device != torch.device("cpu"):
        bce = bce.to(pred.device)
    return bce(pred, target) + dice_loss(pred, target)

# ------------------------------
# Metrics
# ------------------------------
def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def dice_coeff(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2 * intersection + 1e-6) / \
           (pred.sum() + target.sum() + 1e-6)

def accuracy_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    correct = (pred == target).float().sum()
    return correct / torch.numel(pred)

# ------------------------------
# Optimizer
# ------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ------------------------------
# Training Loop
# ------------------------------
epochs = int(os.getenv("EPOCHS", "5"))
best_iou = 0
print("Starting training loop...")

for epoch in range(epochs):
    print(f"\nEpoch [{epoch+1}/{epochs}]")
    model.train()
    train_loss = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)

    # --------------------------
    # Validation
    # --------------------------
    print(f"  Validating...")
    model.eval()
    val_iou = 0
    val_dice = 0
    val_acc = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            val_iou += iou_score(preds, masks)
            val_dice += dice_coeff(preds, masks)
            val_acc += accuracy_score(preds, masks)

    val_iou /= len(val_loader)
    val_dice /= len(val_loader)
    val_acc /= len(val_loader)

    print(f"Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | Val Acc: {val_acc:.4f}")

    # Save Best Model
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), os.path.join(script_dir, "best_model.pth"))

# Save the final model
torch.save(model.state_dict(), os.path.join(script_dir, "best_model.pth"))

# ------------------------------
# Test Evaluation
# ------------------------------
print("\n" + "="*50)
print("Evaluating on test set...")
model.load_state_dict(torch.load(os.path.join(script_dir, "best_model.pth")))
model.eval()

test_iou = 0
test_dice = 0
test_acc = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        test_iou += iou_score(preds, masks)
        test_dice += dice_coeff(preds, masks)
        test_acc += accuracy_score(preds, masks)

test_iou /= len(test_loader)
test_dice /= len(test_loader)
test_acc /= len(test_loader)

print("===== Test Results =====")
print(f"Test IoU: {test_iou:.4f}")
print(f"Test Dice: {test_dice:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("Training completed successfully!")
