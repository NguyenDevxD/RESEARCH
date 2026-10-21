import os
from datetime import datetime
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torchvision import transforms

DATA_DIR = r"D:\Dung Project\dungvenv\research"

DATASET_NAME = "dataset_full"
DATA_CSV = os.path.join(DATA_DIR, r"caxton_dataset\full_data.csv")
DATASET_MEAN = [0.288, 0.266, 0.137]
DATASET_STD = [0.196, 0.194, 0.173]

INITIAL_LR = 1e-3

BATCH_SIZE = 32
MAX_EPOCHS = 40

NUM_NODES = 1
NUM_GPUS = 1
ACCELERATOR = "gpu"


def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_dirs(path):
    try:
        os.makedirs(path)
    except:
        pass

preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            DATASET_MEAN,
            DATASET_STD,
        )
    ],
)