import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar

from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier
from train_config import *


def main():

    # -----------------------------
    # Argument parser
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=1234, type=int)
    parser.add_argument("-e", "--epochs", default=MAX_EPOCHS, type=int)

    args = parser.parse_args()

    seed = args.seed
    set_seed(seed)

    # -----------------------------
    # Paths & Logger
    # -----------------------------
    logs_dir = f"logs/{DATASET_NAME}_classifier_{seed}/"
    os.makedirs(logs_dir, exist_ok=True)

    tb_logger = pl_loggers.TensorBoardLogger(logs_dir)

    checkpoint_dir = f"checkpoints/{DATASET_NAME}_classifier_{seed}/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # -----------------------------
    # Callbacks
    # -----------------------------
    checkpoint_best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_acc_adhesion",
        mode="max",
        save_top_k=1,
        filename="epoch{epoch:02d}-val_acc_adhesion{val_acc_adhesion:.4f}",
        auto_insert_metric_name=False
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    rich_progress = RichProgressBar(refresh_rate=1)

    callbacks = [
        checkpoint_best,
        lr_monitor,
        rich_progress
    ]

    # -----------------------------
    # Checkpoint path
    # -----------------------------
    CHECKPOINT_PATH = r"D:\Dung Project\dungvenv\research\epoch01-val_acc0.0000.ckpt"

    # -----------------------------
    # Model
    # -----------------------------
    model = ParametersClassifier(
        lr=INITIAL_LR,
        max_epochs=args.epochs,
    )

    if CHECKPOINT_PATH is not None and os.path.exists(CHECKPOINT_PATH):

        print(f"\n🔄 Loading BACKBONE ONLY from:\n{CHECKPOINT_PATH}")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        backbone_state_dict = {
            k.replace("attention_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("attention_model.")
        }

        model.attention_model.load_state_dict(backbone_state_dict, strict=False)

        print("✅ Backbone loaded successfully")
        print("🔥 Mode: Train backbone + adhesion (freeze 2 heads)\n")

    else:
        print("\n🚀 Training from scratch (backbone + adhesion)\n")

    # -----------------------------
    # Data
    # -----------------------------
    data = ParametersDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        csv_file=DATA_CSV,
        dataset_name=DATASET_NAME,
        mean=DATASET_MEAN,
        std=DATASET_STD,
    )

    data.setup()

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = pl.Trainer(

        max_epochs=args.epochs,

        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=NUM_GPUS if torch.cuda.is_available() else 1,

        precision="16-mixed",

        gradient_clip_val=1.0,

        callbacks=callbacks,
        logger=tb_logger,

        log_every_n_steps=20,

        deterministic=True
    )

    # -----------------------------
    # Train
    # -----------------------------
    print(f"🚀 Start training (seed={seed})")
    print("🎯 Objective: Improve adhesion with backbone adaptation\n")

    trainer.fit(model, datamodule=data)

    print("\n✅ Training completed!")
    print(f"📌 Checkpoints saved at: {checkpoint_dir}")


if __name__ == "__main__":
    main()