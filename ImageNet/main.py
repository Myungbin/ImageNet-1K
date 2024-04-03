import warnings

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ImageNet.config.config import CFG
from ImageNet.data.dataloader import DatasetLoader
from ImageNet.model.models import VisionTransformer
from ImageNet.train.trainer import Trainer
from ImageNet.utils.logging import set_logging

warnings.filterwarnings(action="ignore")


if __name__ == "__main__":
    set_logging()
    dataloader = DatasetLoader(num_workers=CFG.NUM_WORKERS)
    train_loader, val_loader = dataloader.load
    model = VisionTransformer(num_classes=CFG.NUM_CLASS)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0, verbose=True)
    trainer = Trainer(model, criterion, optimizer, scheduler, scaler, True)
    trainer.fit(train_loader, val_loader)
    ...