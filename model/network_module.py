import torch
import torch.nn as nn
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl

from model.residual_attention_network import ResidualAttentionModel_56


class ParametersClassifier(pl.LightningModule):

    def __init__(
        self,
        lr=5e-5,
        max_epochs=60,
        transfer=False,
        trainable_layers=1,
        retrieve_layers=False,
        retrieve_masks=False,
        pos_weight=None
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.max_epochs = max_epochs

        # ================= Loss weight =================
        self.adhesion_weight = 2.5

        # ================= Loss function =================
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # ================= Backbone =================
        self.attention_model = ResidualAttentionModel_56(
            retrieve_layers=retrieve_layers,
            retrieve_masks=retrieve_masks
        )

        num_ftrs = 2048

        # ================= Classification heads =================
        def build_head():
            return nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(512, 128),
                nn.ReLU(),

                nn.Linear(128, 1)
            )

        self.fc_uniformity = build_head()
        self.fc_continuity = build_head()
        self.fc_adhesion = build_head()

        # ================= Transfer learning =================
        if transfer:
            for child in list(self.attention_model.children())[:-trainable_layers]:
                for param in child.parameters():
                    param.requires_grad = False

        # ================= Loss =================
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    # ================================================= Forward
    def forward(self, x):

        features = self.attention_model(x)

        u = self.fc_uniformity(features)
        c = self.fc_continuity(features)
        a = self.fc_adhesion(features)

        logits = torch.cat([u, c, a], dim=1)

        return logits

    # ================================================= Loss
    def compute_loss(self, logits, y):

        y = y.float()

        loss_u = self.criterion(logits[:, 0], y[:, 0])
        loss_c = self.criterion(logits[:, 1], y[:, 1])
        loss_a = self.criterion(logits[:, 2], y[:, 2])

        loss = loss_u + loss_c + self.adhesion_weight * loss_a

        return loss

    # ================================================= Accuracy
    def compute_accuracy(self, logits, y):

        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).long()

        acc_u = (preds[:, 0] == y[:, 0]).float().mean()
        acc_c = (preds[:, 1] == y[:, 1]).float().mean()
        acc_a = (preds[:, 2] == y[:, 2]).float().mean()

        acc = (acc_u + acc_c + acc_a) / 3

        return acc, acc_u, acc_c, acc_a

    # ================================================= Training
    def training_step(self, batch, batch_idx):

        x, y, _ = batch

        logits = self(x)

        loss = self.compute_loss(logits, y)
        acc, acc_u, acc_c, acc_a = self.compute_accuracy(logits, y)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)

        self.log("train_acc_uniformity", acc_u)
        self.log("train_acc_continuity", acc_c)
        self.log("train_acc_adhesion", acc_a)

        return loss

    # ================================================= Validation
    def validation_step(self, batch, batch_idx):

        x, y, _ = batch

        logits = self(x)

        loss = self.compute_loss(logits, y)
        acc, acc_u, acc_c, acc_a = self.compute_accuracy(logits, y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        self.log("val_acc_uniformity", acc_u, sync_dist=True, on_epoch= True, prog_bar= True)
        self.log("val_acc_continuity", acc_c, sync_dist=True, on_epoch= True, prog_bar= True)
        self.log("val_acc_adhesion", acc_a, sync_dist=True, on_epoch= True, prog_bar=True)

        return loss

    # ================================================= Test
    def test_step(self, batch, batch_idx):

        x, y, _ = batch

        logits = self(x)

        loss = self.compute_loss(logits, y)
        acc, acc_u, acc_c, acc_a = self.compute_accuracy(logits, y)

        self.log("test_loss", loss)
        self.log("test_acc", acc)

        self.log("test_acc_uniformity", acc_u)
        self.log("test_acc_continuity", acc_c)
        self.log("test_acc_adhesion", acc_a)

        return loss

    # ================================================= Optimizer
    def configure_optimizers(self):

        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )

        total_steps = self.trainer.estimated_stepping_batches

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.1,        # 10% training để tăng LR
            anneal_strategy="cos",
            div_factor=25,        # lr_start = lr / 25
            final_div_factor=100  # lr_end = lr / 100
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }