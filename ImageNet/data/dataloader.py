import logging

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from ImageNet.config.config import CFG


class DatasetLoader:
    def __init__(self, num_workers=0, shuffle=True) -> None:
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.train_dataset = ImageFolder(CFG.TRAIN_PATH, transform=self.train_transform)
        self.val_dataset = ImageFolder(CFG.TEST_PATH, transform=self.val_transform)

        logging.info("Dataset Info:")
        logging.info("------------------------------------------------------------")
        logging.info(f"Train Data Size: {len(self.train_dataset)}")
        logging.info(f"Validation Data Size: {len(self.val_dataset)}")
        logging.info(f"Image Trasform: {self.train_transform}")

    @property
    def load(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=self.shuffle,
            batch_size=CFG.BATCH_SIZE,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        val_dataloader = DataLoader(
            self.val_dataset,
            shuffle=self.shuffle,
            batch_size=CFG.BATCH_SIZE,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        return train_dataloader, val_dataloader


if __name__ == "__main__":
    dataloader = DatasetLoader(num_workers=CFG.NUM_WORKERS)
    train_loader, val_loader = dataloader.load

    for i in train_loader:
        print(i)
        break
