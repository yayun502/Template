# Main change for this branch
- train 支援 離線載入 resnet50 權重
- train 支援 輸出 logging csv
- infer 改為 整個 test split inference  
- infer 支援 輸出 result csv (+ confusion matrix 圖檔)
# Overview
## 核心設計
- 同一個 defect sample 有多張子圖
- 依 FOV 分成 local branch 與 global branch
- 各分支做 attention pooling
- 最後進行 sample-level 4-class classification
- 同時做 attribute multi-task learning

## 資料格式
```
dataset/
├── train/
├── val/
└── test/
```
每個 sample 資料夾包含：
- label.txt
- meta.json
- 多張 png/jpg 影像
  
# Architecture
```
Project/
├── configs/
│   └── config.py
├── data/
│   └── dataset.py
├── models/
│   └── defect_model.py
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   └── seed.py
├── train.py
├── infer.py
├── requirements.txt
└── README.md
```
# Data Structure
```
dataset/
├── train/
│   ├── sample_0001/
│   │   ├── label.txt
│   │   ├── meta.json
│   │   ├── img_01.png
│   │   ├── img_02.png
│   │   └── img_03.png
│   ├── sample_0002/
│   └── ...
├── val/
│   ├── sample_1001/
│   └── ...
└── test/
    ├── sample_2001/
    └── ...
```
### label.txt
```
NP
```
### meta.json
```
{
  "images": [
    {"file": "img_01.png", "fov": 12},
    {"file": "img_02.png", "fov": 18},
    {"file": "img_03.png", "fov": 35}
  ]
}
```

# Train
```bash
python train.py
```
# Inference
```
python infer.py --sample_dir ./dataset/val/sample_1001 --ckpt ./checkpoints/best_model.pt
```
