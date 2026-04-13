import os

# ========= Dataset =========
DATA_ROOT = "./dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# ========= Labels =========
LABEL_MAP = {
    "Single": 0,
    "NP": 1,
    "定點": 2,
    "Multi": 3
}

IDX2LABEL = {v: k for k, v in LABEL_MAP.items()}

# ========= Image / Loader =========
IMAGE_SIZE = 224
MAX_LOCAL_VIEWS = 6
MAX_GLOBAL_VIEWS = 6
LOCAL_FOV_THRESHOLD = 29

NUM_WORKERS = 4
BATCH_SIZE = 8

# ========= Model =========
BACKBONE_NAME = "convnext_tiny"
FEAT_DIM = 256
NUM_CLASSES = 4
NUM_ATTRS = 4   # has_np, repetitive, breakpoint, single_structure

# ========= Training =========
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
ATTR_LOSS_WEIGHT = 0.5
DEVICE = "cuda"

# ========= Save =========
SAVE_DIR = "./checkpoints"
BEST_MODEL_NAME = "best_model.pt"
LAST_MODEL_NAME = "last_model.pt"
