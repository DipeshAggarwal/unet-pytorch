import torch
import os

DATASET_PATH = os.path.join("dataset")

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASKS_DATASET_PATH = os.path.join(DATASET_PATH, "masks")

TEST_SPLIT = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

NUM_CHANNELS = 1
NUM_CLASSES = 1
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count()

LR = 1e-3
NUM_EPOCHS = 40
BATCH_SIZE = 64

INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128

# Threshold to filter weak predictions
THRESHOLD = 0.5

BASE_OUTPUT = "output"

MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.model")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")
TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")
