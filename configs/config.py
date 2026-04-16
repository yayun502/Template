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
# 可選: "resnet50_local" 或 "timm"
BACKBONE_TYPE = "resnet50_local"

# 如果是 timm 模式才會用到
BACKBONE_NAME = "convnext_tiny"

# 如果是本地 resnet50 權重模式，請填你的 .pth 路徑
LOCAL_PRETRAINED_PATH = "./pretrained/resnet50.pth"

FEAT_DIM = 256
NUM_CLASSES = 4

# ========= Training =========
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"

# ========= Class Weight =========
# 順序必須和 LABEL_MAP 一致:
# 0: Single, 1: NP, 2: 定點, 3: Multi
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS = [1.0, 2.0, 2.0, 1.5]

# ========= Main Classification Loss =========
# 可選: "ce" 或 "focal"
CLS_LOSS_TYPE = "focal"
FOCAL_GAMMA = 2.0

# ========= Hierarchical Head =========
USE_HIERARCHICAL_HEAD = True
HIER_LOSS_WEIGHT = 0.5
GATE_LOSS_WEIGHT = 0.3

# inference 時主預測使用哪個 head:
# 可選: "main" 或 "hier"
INFER_PRED_HEAD = "main"

# ========= Scheduler =========
# 可選: "none", "cosine", "step", "plateau"
SCHEDULER_TYPE = "cosine"

# for StepLR
STEP_SIZE = 10
STEP_GAMMA = 0.1

# for ReduceLROnPlateau
PLATEAU_MODE = "max"
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 3

# for CosineAnnealingLR
COSINE_T_MAX = EPOCHS
COSINE_ETA_MIN = 1e-6

# ========= Save =========
SAVE_DIR = "./checkpoints"
BEST_MODEL_NAME = "best_model.pt"
LAST_MODEL_NAME = "last_model.pt"

# ========= Logging =========
LOG_DIR = "./logs"
TRAIN_LOG_CSV = os.path.join(LOG_DIR, "train_log.csv")

# ========= Inference =========
INFER_DIR = "./inference_outputs"
TEST_PRED_CSV = os.path.join(INFER_DIR, "test_predictions.csv")
TEST_CM_PNG = os.path.join(INFER_DIR, "test_confusion_matrix.png")

ATTN_CSV = os.path.join(INFER_DIR, "test_attention_weights.csv")
ATTN_FIG_DIR = os.path.join(INFER_DIR, "attention_figures")
