import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import ImageFile, Image
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParametersDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        image_dim=(320, 320),
        pre_crop_transform=None,
        post_crop_transform=None,
        classification=True,
        uniformity=True,
        continuity=True,
        adhesion=True,
        per_img_normalisation=False,
    ):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.pre_crop_transform = pre_crop_transform
        self.post_crop_transform = post_crop_transform
        self.image_dim = image_dim

        # bật/tắt nhãn
        self.use_uniformity = uniformity
        self.use_continuity = continuity
        self.use_adhesion = adhesion

        self.per_img_normalisation = per_img_normalisation
        self.classification = classification

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # ====== Load image ======
        img_name = os.path.join(self.root_dir, self.dataframe.img_path[idx])

        dim = self.image_dim[0] / 2
        left = self.dataframe.nozzle_tip_x[idx] - dim
        top = self.dataframe.nozzle_tip_y[idx] - dim
        right = self.dataframe.nozzle_tip_x[idx] + dim
        bottom = self.dataframe.nozzle_tip_y[idx] + dim

        image = Image.open(img_name).convert("RGB")

        if self.pre_crop_transform:
            image = self.pre_crop_transform(image)

        image = image.crop((left, top, right, bottom))

        # ====== Chuẩn hoá ảnh ======
        if self.per_img_normalisation:
            tfms = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor()
            ])

            image = tfms(image)

            mean = torch.mean(image, dim=[1, 2])
            std = torch.std(image, dim=[1, 2])

            image = transforms.Normalize(mean, std)(image)

        else:
            if self.post_crop_transform:
                image = self.post_crop_transform(image)

        # ====== Labels ======
        targets = []

        if self.use_uniformity:
            targets.append(int(self.dataframe.uniformity[idx]))

        if self.use_continuity:
            targets.append(int(self.dataframe.continuity[idx]))

        if self.use_adhesion:
            targets.append(int(self.dataframe.adhesion[idx]))

        y = torch.tensor(targets, dtype=torch.long)

        return image, y, img_name