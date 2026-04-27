import os

DATA_ROOT = "./dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

LABEL_MAP = {
    "Single": 0,
    "NP": 1,
    "定點": 2,
    "Multi": 3
}
IDX2LABEL = {v: k for k, v in LABEL_MAP.items()}

IMAGE_SIZE = 518
MAX_LOCAL_VIEWS = 6
MAX_GLOBAL_VIEWS = 6
LOCAL_FOV_THRESHOLD = 29

NUM_WORKERS = 4
BATCH_SIZE = 8

# ========= Backbone =========
# "resnet50_local" / "dinov2_local" / "timm"
BACKBONE_TYPE = "dinov2_local"
BACKBONE_NAME = "dinov2_vitb14"
DINO_REPO_DIR = "./third_party/dinov2"
LOCAL_PRETRAINED_PATH = "./pretrained/dinov2_vitb14_pretrain.pth"

FEAT_DIM = 256
NUM_CLASSES = 4

# ========= Training =========
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"

# ========= Class Weight =========
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS = [1.0, 2.0, 2.0, 1.5]

# ========= Main Classification Loss =========
CLS_LOSS_TYPE = "focal"   # "ce" or "focal"
FOCAL_GAMMA = 2.0

# ========= Hierarchical Head =========
USE_HIERARCHICAL_HEAD = True
HIER_LOSS_WEIGHT = 0.5
GATE_LOSS_WEIGHT = 0.3
INFER_PRED_HEAD = "main"  # "main" or "hier"

# ========= Branch Gate =========
USE_BRANCH_GATE = True

# "direct": local_w * local_feat, global_w * global_feat
# "residual": (0.5 + local_w) * local_feat, (0.5 + global_w) * global_feat
BRANCH_GATE_MODE = "residual"

# entropy regularization:
# 若啟用，會鼓勵 branch gate 不要太早 collapse 到單邊
USE_BRANCH_ENTROPY_REG = False
BRANCH_ENTROPY_WEIGHT = 0.01

# ========= Scheduler =========
SCHEDULER_TYPE = "cosine"  # "none", "cosine", "step", "plateau"

STEP_SIZE = 10
STEP_GAMMA = 0.1

PLATEAU_MODE = "max"
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 3

COSINE_T_MAX = EPOCHS
COSINE_ETA_MIN = 1e-6

SAVE_DIR = "./checkpoints"
BEST_MODEL_NAME = "best_model.pt"
LAST_MODEL_NAME = "last_model.pt"

LOG_DIR = "./logs"
TRAIN_LOG_CSV = os.path.join(LOG_DIR, "train_log.csv")

INFER_DIR = "./inference_outputs"
TEST_PRED_CSV = os.path.join(INFER_DIR, "test_predictions.csv")
TEST_CM_PNG = os.path.join(INFER_DIR, "test_confusion_matrix.png")

ATTN_CSV = os.path.join(INFER_DIR, "test_attention_weights.csv")
ATTN_FIG_DIR = os.path.join(INFER_DIR, "attention_figures")
