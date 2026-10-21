import os
import argparse
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier
from train_config import *


def safe_unpack(batch):
    """
    Dataset có thể trả:
      (x, y, path)
      (x, y)
    """
    if len(batch) == 3:
        return batch
    else:
        x, y = batch
        path = ["N/A"] * x.size(0)
        return x, y, path


def main():
    # -----------------------------
    # Parse arguments
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=1234, type=int)
    args = parser.parse_args()
    set_seed(args.seed)

    # -----------------------------
    # Checkpoint
    # -----------------------------
    ckpt_path = r"D:\Dung Project\dungvenv\research\epoch01-val_acc0.0000.ckpt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"⚠️ Checkpoint not found at {ckpt_path}")

    # -----------------------------
    # Load model
    # -----------------------------
    model = ParametersClassifier.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        lr=INITIAL_LR,
    )
    model.eval()

    # -----------------------------
    # Load Data
    # -----------------------------
    data = ParametersDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        csv_file=DATA_CSV,
        image_dim=(320, 320),
        dataset_name=DATASET_NAME,
        mean=DATASET_MEAN,
        std=DATASET_STD,
        transform=False,
    )
    data.setup("test")

    # -----------------------------
    # PL Test
    # -----------------------------
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16,
    )

    print(f"\n🚀 Running test using checkpoint: {ckpt_path}\n")
    trainer.test(model, datamodule=data)

    # -----------------------------
    # Custom Metrics (Accuracy)
    # -----------------------------
    print("\n📊 Calculating per-head accuracy...\n")

    test_loader = data.test_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    correct = torch.zeros(3)
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            x, y, path = safe_unpack(batch)

            x = x.to(device)
            y = y.to(device).long()

            logits = model(x)
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).long()

            correct += (preds == y).sum(dim=0).cpu()
            total += y.size(0)

    acc = correct / total

    print("🎯 Per-Head Accuracy")
    print(f"Uniformity : {acc[0]:.4f}")
    print(f"Continuity : {acc[1]:.4f}")
    print(f"Adhesion   : {acc[2]:.4f}")

    print(f"\n🔥 Mean Accuracy: {acc.mean():.4f}")

    # -----------------------------
    # Visualization
    # -----------------------------
    batch = next(iter(test_loader))
    x, y, path = safe_unpack(batch)

    x = x.to(device)

    with torch.no_grad():
        logits = model(x)

    preds = torch.sigmoid(logits)
    preds = (preds > 0.5).long().cpu()

    y = y.cpu()
    x = x.cpu()

    num_show = min(5, len(x))

    grid = make_grid(x[:num_show], nrow=num_show, normalize=True)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Sample Test Images")
    plt.show()

    print("\n🔍 Sample predictions:")
    for i in range(num_show):
        print(f"🖼️ Image {i+1} | {path[i]}")
        print(f"   Pred : {preds[i].tolist()}")
        print(f"   Label: {y[i].tolist()}")


if __name__ == "__main__":
    main()