import os
from pathlib import Path

import torch


class Config:
    # Directory
    ROOT_PATH = Path(__file__).parents[2]
    TRAIN_PATH = os.path.join(ROOT_PATH, "data", "train")
    TEST_PATH = os.path.join(ROOT_PATH, "data", "val")
    SAVE_MODEL_PATH = os.path.join(ROOT_PATH, "check_points")
    LOG_DIR = os.path.join(ROOT_PATH, "logs")

    # data
    H = 224
    W = 224
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    NUM_CLASS = len(os.listdir(TRAIN_PATH))

    # train
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = torch.cuda.device_count() * 4
    BATCH_SIZE = 32
    EPOCHS = 100
    SEED = 42
    SHUFFLE = True
    LEARNING_RATE = 5e-6


CFG = Config()


if __name__ == "__main__":
    print(str(CFG.ROOT_PATH).split("/")[-1])
