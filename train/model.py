# Module 3: Model Development (Segmentation and Classification)

import torch
import torch.nn as nn

# Double Convolution Block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        bn = self.bottleneck(p3)

        up3 = self.up3(bn)
        up3 = torch.cat([up3, d3], dim=1)
        up3 = self.conv3(up3)

        up2 = self.up2(up3)
        up2 = torch.cat([up2, d2], dim=1)
        up2 = self.conv2(up2)

        up1 = self.up1(up2)
        up1 = torch.cat([up1, d1], dim=1)
        up1 = self.conv1(up1)

        return torch.sigmoid(self.final(up1))
def dice_loss(pred, target, smooth=1):
    """Calculate Dice loss for binary masks.

    The input tensors are flattened so that the computation works for
    predictions of shape ``(N,1,H,W)`` as well as ``(N,H,W)``. A small
    ``smooth`` term avoids division by zero.
    """
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))




def combined_loss(pred, target):
    """Binary cross entropy + dice loss.

    Args:
        pred (Tensor): network output after sigmoid, shape (N,1,H,W) or
            similar.
        target (Tensor): ground‑truth mask with same shape.
    Returns:
        Tensor: scalar loss value.
    """
    # create BCE on the same device as the prediction
    bce = nn.BCELoss()
    if pred.device != torch.device("cpu"):
        bce = bce.to(pred.device)
    return bce(pred, target) + dice_loss(pred, target)
def iou_score(pred, target, threshold=0.5):
    """Intersection‑over‑union metric for binary segmentation.

    ``pred`` is thresholded into a binary mask before computing IoU.
    """
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)
def dice_coeff(pred, target, threshold=0.5):
    """Dice coefficient metric (same as F1 score for masks).

    The prediction is binarized using ``threshold`` and then compared
    against the ground truth.
    """
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2 * intersection + 1e-6) / \
           (pred.sum() + target.sum() + 1e-6)
# ==============================
# TEST BLOCK 
# ==============================

if __name__ == "__main__":

    # Create model
    model = UNet(in_channels=3, out_channels=1)

    # Print model summary
    print(model)

    # Create dummy input (Batch size 1, 3 channels, 256x256)
    x = torch.randn(1, 3, 256, 256)

    # Forward pass
    output = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)