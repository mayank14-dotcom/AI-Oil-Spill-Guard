# Milestone 3 : Week 5-6
# Module 5: Visualization of Results

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
#  MODULE 5: VISUALIZATION OF RESULTS
# ==========================================================

# ----------------------------------------------------------
# Side-by-Side Comparison
# ----------------------------------------------------------

def visualize_predictions(model, dataloader, device, save_dir="results/comparisons", num_samples=10):
    print("[debug] visualize_predictions called", flush=True)
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            print(f"[debug] batch {idx}, batch size {images.size(0)}", flush=True)

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = (outputs > 0.5).float()

            for i in range(images.size(0)):

                image = images[i].cpu().permute(1, 2, 0).numpy()
                mask = masks[i].cpu().squeeze().numpy()
                pred = preds[i].cpu().squeeze().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                axs[0].imshow(image)
                axs[0].set_title("Original Image")
                axs[0].axis("off")

                axs[1].imshow(mask, cmap="gray")
                axs[1].set_title("Ground Truth")
                axs[1].axis("off")

                axs[2].imshow(pred, cmap="gray")
                axs[2].set_title("Prediction")
                axs[2].axis("off")

                plt.tight_layout()
                outpath = f"{save_dir}/comparison_{idx}_{i}.png"
                plt.savefig(outpath)
                plt.close()
                print(f"[debug] saved {outpath}", flush=True)

                if idx * images.size(0) + i >= num_samples:
                    print("[debug] reached num_samples limit, returning", flush=True)
                    return


# ----------------------------------------------------------
# 2 Overlay Function
# ----------------------------------------------------------

def overlay_mask_on_image(image, mask, alpha=0.4):
    """
    Overlay predicted mask on original image using red color.
    """
    if image.max() <= 1:
        image = image * 255

    image = image.astype(np.uint8)

    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = mask * 255  # Red channel for oil spill

    overlay = image * (1 - alpha) + colored_mask * alpha
    return overlay.astype(np.uint8)


# ----------------------------------------------------------
#  Overlay Visualization
# ----------------------------------------------------------

def visualize_overlay(model, dataloader, device, save_dir="results/overlays", num_samples=10):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, _) in enumerate(dataloader):

            images = images.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()

            for i in range(images.size(0)):

                image = images[i].cpu().permute(1, 2, 0).numpy()
                pred = preds[i].cpu().squeeze().numpy()

                overlay = overlay_mask_on_image(image, pred)

                plt.figure(figsize=(6, 6))
                plt.imshow(overlay)
                plt.title("Oil Spill Detection Overlay")
                plt.axis("off")

                plt.savefig(f"{save_dir}/overlay_{idx}_{i}.png")
                plt.close()

                if idx * images.size(0) + i >= num_samples:
                    return


# ----------------------------------------------------------
#  Summary Grid for Reports
# ----------------------------------------------------------

def create_summary_grid(model, dataloader, device, save_path="results/summary_grid.png", show_plot=False):
    """Generate and optionally display a small grid of example predictions.

    Args:
        model: trained segmentation network
        dataloader: iterable returning (image, mask) pairs
        device: torch.device for computation
        save_path: where to write the resulting PNG
        show_plot: if True, display the figure via ``plt.show()`` after saving
    """
    model.eval()
    os.makedirs("results", exist_ok=True)

    with torch.no_grad():
        images, masks = next(iter(dataloader))

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        preds = (outputs > 0.5).float()

        # determine how many samples we actually have (batch may be smaller than 4)
        batch_n = images.size(0)
        rows = min(4, batch_n)

        fig, axs = plt.subplots(rows, 3, figsize=(12, 4 * rows))
        # ensure axs is 2‑D for consistent indexing
        if rows == 1:
            axs = axs[np.newaxis, :]

        for i in range(rows):

            image = images[i].cpu().permute(1, 2, 0).numpy()
            mask = masks[i].cpu().squeeze().numpy()
            pred = preds[i].cpu().squeeze().numpy()

            axs[i, 0].imshow(image)
            axs[i, 0].set_title("Original")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(mask, cmap="gray")
            axs[i, 1].set_title("Ground Truth")
            axs[i, 1].axis("off")

            axs[i, 2].imshow(pred, cmap="gray")
            axs[i, 2].set_title("Prediction")
            axs[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        if show_plot:
            plt.show()
        plt.close(fig)


# ----------------------------------------------------------
#  Complete Execution Function
# ----------------------------------------------------------

def run_visualization_pipeline(model, test_loader, device, display=False):
    """Run all three visualization steps.

    If ``display`` is True the final summary grid will also be shown
    interactively (useful when running the script directly).
    """

    print("Generating side-by-side comparisons...")
    visualize_predictions(model, test_loader, device)

    print("Generating overlay visualizations...")
    visualize_overlay(model, test_loader, device)

    print("Generating summary grid...")
    create_summary_grid(model, test_loader, device, show_plot=display)

    print("Visualization Complete. Results saved in 'results/' folder.")


# ==========================================================
#  CALL THIS AFTER TRAINING
# ==========================================================
if __name__ == "__main__":
    # Try to import UNet from the local model module (same folder).
    try:
        from model import UNet
    except Exception:
        try:
            from train.model import UNet
        except Exception:
            UNet = None

    if UNet is None:
        print("UNet model class not found. To run visualizations, ensure `UNet` is importable (from model import UNet).")
        print("Alternatively, call run_visualization_pipeline(model, test_loader, device, display=True) from your training script.")
    else:
        model_path = "model.pth"
        model = UNet(in_channels=3, out_channels=1)

        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                print(f"Loaded model weights from {model_path}")
            except Exception as e:
                print(f"Warning: failed to load model weights: {e}")
        else:
            print(f"Model file {model_path} not found; using fresh UNet instance.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Check whether a `test_loader` exists in this namespace; if not, attempt to create one.
        try:
            test_loader  # reference to see if defined elsewhere
            has_loader = True
        except NameError:
            has_loader = False

        if not has_loader:
            # Try to build a small test_loader using the project's `dataset.OilDataset`.
            try:
                from dataset import OilDataset
                from torchvision import transforms
                from torchvision.transforms import InterpolationMode, PILToTensor
                from torch.utils.data import DataLoader, Subset

                script_dir = os.path.dirname(os.path.abspath(__file__))
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])
                mask_transform = transforms.Compose([
                    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
                    PILToTensor(),
                ])

                full_dataset = OilDataset(
                    script_dir,
                    script_dir,
                    transform=transform,
                    mask_transform=mask_transform,
                )
                n = len(full_dataset)
                if n == 0:
                    raise RuntimeError(f"No images found in {script_dir} to create a test loader.")

                # take up to 4 samples for quick visualization (smaller batches finish faster)
                k = min(4, n)
                indices = list(range(n))[-k:]
                test_dataset = Subset(full_dataset, indices)
                test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

                print(f"Created test_loader from dataset with {len(test_dataset)} samples.")
                run_visualization_pipeline(model, test_loader, device, display=True)

            except Exception as e:
                print(f"Could not auto-create test_loader: {e}")
                print("Please provide a DataLoader named `test_loader` or run this from your training script.")
        else:
            run_visualization_pipeline(model, test_loader, device, display=True)
