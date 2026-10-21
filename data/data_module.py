import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import torch
from PIL import ImageFile
from data.dataset import ParametersDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParametersDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        data_dir,
        csv_file,
        dataset_name,
        mean,
        std,
        load_saved=False,
        transform=True,
        image_dim=(320, 320),
        per_img_normalisation=False,
        classification=True,
        uniformity=True,
        continuity=True,
        adhesion=True,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.mean = mean
        self.std = std
        self.transform = transform
        self.classification = classification

        if self.transform:
            self.pre_crop_transform = transforms.Compose(
                [
                    transforms.RandomRotation(5),
                    transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
                ]
            )

            self.post_crop_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        hue=0.1,
                        saturation=0.1,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )

        else:
            self.pre_crop_transform = None

            self.post_crop_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )

        self.dims = (3, 224, 224)

        # multi-label → 3 outputs
        self.num_classes = 3

        self.load_saved = load_saved
        self.image_dim = image_dim
        self.per_img_normalisation = per_img_normalisation

        # geometric labels
        self.use_uniformity = uniformity
        self.use_continuity = continuity
        self.use_adhesion = adhesion

    def setup(self, stage=None, save=False, test_all=False):

        self.dataset = ParametersDataset(
            csv_file=self.csv_file,
            root_dir=self.data_dir,
            image_dim=self.image_dim,
            pre_crop_transform=self.pre_crop_transform,
            post_crop_transform=self.post_crop_transform,
            classification=self.classification,
            uniformity=self.use_uniformity,
            continuity=self.use_continuity,
            adhesion=self.use_adhesion,
            per_img_normalisation=self.per_img_normalisation,
        )

        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.2 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        if save:
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = torch.utils.data.random_split(
                self.dataset, [train_size, val_size, test_size]
            )

            os.makedirs(f"data/{self.dataset_name}/", exist_ok=True)

            torch.save(self.train_dataset, f"data/{self.dataset_name}/train.pt")
            torch.save(self.val_dataset, f"data/{self.dataset_name}/val.pt")
            torch.save(self.test_dataset, f"data/{self.dataset_name}/test.pt")

        if stage == "fit" or stage is None:

            if self.load_saved:
                self.train_dataset = torch.load(
                    f"data/{self.dataset_name}/train.pt"
                )
                self.val_dataset = torch.load(
                    f"data/{self.dataset_name}/val.pt"
                )
            else:
                self.train_dataset, self.val_dataset, _ = torch.utils.data.random_split(
                    self.dataset, [train_size, val_size, test_size]
                )

        if stage == "test" or stage is None:

            if self.load_saved:
                self.test_dataset = torch.load(
                    f"data/{self.dataset_name}/test.pt"
                )

            else:
                if test_all:
                    self.test_dataset = self.dataset
                else:
                    _, _, self.test_dataset = torch.utils.data.random_split(
                        self.dataset, [train_size, val_size, test_size]
                    )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=7,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=7,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=7,
            pin_memory=True,
        )