# Wafer Edge Pipeline v3

這份工具用來處理 wafer 邊緣的細小缺陷，例如：

- Lowfill
- Overfill
- Splash
- Edge crack
- Edge contamination

它會把圓形 wafer 邊緣拉直，再切成小圖供 YOLO 訓練。

---

# 1. v3 與舊版差異

v3 不會先把所有異常合成一張單通道 mask。

而是：

```text
每一個標註 instance
        ↓
各自轉成 mask
        ↓
各自做極座標展開
        ↓
各自切 patch
        ↓
最後輸出 YOLO label
```

因此可以：

- 同一片 wafer 有多種異常
- 同一個 patch 有多種異常
- 相同位置有重疊標註
- 保留每個 instance，不會因 class mask 覆蓋而遺失

---

# 2. 支援的標註格式

| 原始格式 | annotation_format |
|---|---|
| Labelme JSON | `labelme` |
| YOLO Bounding Box | `yolo_bbox` |
| Mask PNG | `mask` |
| COCO JSON | `coco` |
| 無標註正常片 | `none` |

---

# 3. 三類異常設定

在 `config.yaml`：

```yaml
names:
  - Lowfill
  - Overfill
  - Splash
```

對應：

```text
YOLO class 0 = Lowfill
YOLO class 1 = Overfill
YOLO class 2 = Splash
```

同一張 wafer 可以有：

```text
2 個 Lowfill
1 個 Overfill
3 個 Splash
```

也可以同一個 patch 同時出現多種異常。

---

# 4. 專案結構

```text
wafer_edge_pipeline_v3/
├── wafer_edge_pipeline/
│   ├── annotations/
│   │   ├── common.py
│   │   ├── factory.py
│   │   └── loaders.py
│   ├── exporters/
│   │   └── yolo.py
│   ├── geometry/
│   │   ├── polar.py
│   │   └── transform.py
│   ├── utils/
│   │   ├── geometry_loader.py
│   │   └── report.py
│   ├── config.py
│   └── types.py
├── preprocess.py
├── analyze_dataset.py
├── train.py
├── config.yaml
├── requirements.txt
└── README.md
```

---

# 5. 安裝

```bash
pip install -r requirements.txt
```

公司離線環境請使用內部 wheel 或現有環境。

---

# 6. 資料集結構

先以 wafer 為單位分好 train、val、test。

```text
input_dataset/
├── train/
│   ├── images/
│   └── annotations/
├── val/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/
```

重要：

```text
同一片 wafer 的資料不可以同時出現在 train 和 val。
```

---

# 7. Labelme 資料擺放

config：

```yaml
annotation_format: labelme
task: segment
```

資料：

```text
input_dataset/
├── train/
│   ├── images/
│   │   ├── wafer_001.png
│   │   └── wafer_002.png
│   └── annotations/
│       ├── wafer_001.json
│       └── wafer_002.json
├── val/
└── test/
```

Labelme 中的 label 名稱必須完全一致：

```text
Lowfill
Overfill
Splash
```

同一份 JSON 可以有任意數量的 polygon、rectangle 或 circle。

---

# 8. YOLO Bounding Box 資料擺放

config：

```yaml
annotation_format: yolo_bbox
task: detect
```

資料：

```text
input_dataset/train/
├── images/
│   └── wafer_001.png
└── annotations/
    └── wafer_001.txt
```

`wafer_001.txt`：

```text
0 0.214 0.325 0.020 0.015
1 0.552 0.180 0.035 0.022
2 0.783 0.624 0.018 0.030
0 0.305 0.426 0.025 0.020
```

代表：

- 2 個 Lowfill
- 1 個 Overfill
- 1 個 Splash

---

# 9. Mask PNG 資料擺放

config：

```yaml
annotation_format: mask
task: segment
```

資料：

```text
input_dataset/train/
├── images/
│   └── wafer_001.png
└── annotations/
    └── wafer_001.png
```

Mask value：

```text
0 = Background
1 = Lowfill
2 = Overfill
3 = Splash
```

注意：

Mask 輸入本身仍是單通道，因此無法表示同一 pixel 同時屬於兩種類別。

若需要重疊標註，請使用 Labelme 或 COCO polygon。

---

# 10. COCO 資料擺放

config：

```yaml
annotation_format: coco
task: segment
```

資料：

```text
input_dataset/
├── train/
│   └── images/
├── val/
│   └── images/
├── test/
│   └── images/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

category name 必須是：

```text
Lowfill
Overfill
Splash
```

---

# 11. 正常 wafer

正常 wafer 仍要放進 `images/`。

建議：

- YOLO bbox：建立空 `.txt`
- Labelme：可以沒有 `.json`
- Mask：建立全黑 mask
- COCO：該圖片沒有 annotation

程式會將沒有標註的 wafer 視為正常。

---

# 12. Wafer 幾何設定

```yaml
center_x: 2048
center_y: 2048
inner_radius: 1800
outer_radius: 2020
```

保留範圍應包含：

```text
wafer 內側
+ 完整膠路
+ 外側少量背景
```

不要切得只剩膠線本身。

每張 wafer 圓心不同時：

```yaml
center_json: wafer_geometry.json
```

可參考：

```text
wafer_geometry_example.json
```

---

# 13. 展開與切圖設定

```yaml
unwrap_width: 8192
unwrap_height: 256

patch_width: 512
stride: 384
```

Overlap：

```text
512 - 384 = 128 pixels
```

---

# 14. 執行前處理

```bash
python preprocess.py --config config.yaml --reset
```

輸出：

```text
wafer_edge_dataset/
├── images/
├── labels/
├── metadata/
├── debug/
├── reports/
├── dataset.yaml
└── summary.json
```

---

# 15. Debug 圖

檢查：

```text
wafer_edge_dataset/debug/train/
```

會看到：

```text
wafer_001_unwrapped_instances.png
```

確認：

- wafer 邊緣有完整展開
- 標註位置正確
- 標註沒有上下錯位
- 異常沒有被裁掉

---

# 16. Dataset 統計

前處理完成後：

```bash
python analyze_dataset.py
```

輸出：

```text
wafer_edge_dataset/reports/
├── train_report.json
├── train_class_counts.csv
├── val_report.json
├── val_class_counts.csv
├── test_report.json
├── test_class_counts.csv
└── class_distribution.png
```

可檢查：

- 原始 wafer 數量
- positive patch 數量
- negative patch 數量
- Lowfill 數量
- Overfill 數量
- Splash 數量

---

# 17. 訓練 Segmentation

```bash
python train.py \
  --data wafer_edge_dataset/dataset.yaml \
  --model yolov8m-seg.pt \
  --epochs 100 \
  --imgsz 512 \
  --batch 16
```

---

# 18. 訓練 Detection

先把 config 改成：

```yaml
task: detect
```

重新前處理：

```bash
python preprocess.py --config config.yaml --reset
```

再訓練：

```bash
python train.py \
  --model yolov8m.pt \
  --epochs 100 \
  --imgsz 512 \
  --batch 16
```

---

# 19. 建議第一次測試

先放少量資料：

```text
train：2～5 片
val：1～2 片
```

執行：

```bash
python preprocess.py --config config.yaml --reset
```

優先檢查：

```text
wafer_edge_dataset/debug/
wafer_edge_dataset/reports/
```

確認沒問題後再跑完整資料。

---

# 20. 已知限制

1. Mask PNG 本身不能表示重疊 class。
2. COCO RLE segmentation 尚未支援，只支援 polygon 或 bbox。
3. 推論映射工具尚未包含在此精簡版 v3。
4. 正式產線建議另外加入 wafer-level escape / overkill 報表。
