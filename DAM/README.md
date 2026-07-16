# Wafer Edge Glue Defect Pipeline

流程：

1. 找到 wafer 圓心與邊緣半徑
2. 將邊緣環帶展開成長條圖
3. 沿圓周方向切重疊 patch
4. 將 mask 自動轉成 YOLO segmentation 或 detection labels
5. 使用 Ultralytics YOLO 訓練
6. 推論後將結果映射回原始 wafer

## 1. 安裝

```bash
pip install opencv-python numpy pyyaml ultralytics
```

公司離線環境請使用內部 wheel 或既有 Python environment。

## 2. 輸入資料結構

```text
input_dataset/
├── train/
│   ├── images/
│   │   ├── wafer_001.png
│   │   └── wafer_002.png
│   └── masks/
│       ├── wafer_001.png
│       └── wafer_002.png
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Mask 必須是單通道 PNG：

```text
0 = background
1 = class 0
2 = class 1
...
```

只有一種點膠異常時：

```text
0 = 正常背景
1 = 點膠異常
```

沒有異常的 wafer 可以：

- 放一張全黑 mask，或
- 不放 mask；程式會視為全背景。

## 3. 先修改 config.yaml

至少確認：

```yaml
center_x: 2048
center_y: 2048
inner_radius: 1800
outer_radius: 2020
```

環帶要包含：

- wafer 邊緣內側
- 完整膠路
- wafer 外側少量背景

## 4. 產生 YOLO 資料集

```bash
python preprocess.py --config config.yaml --reset
```

輸出：

```text
wafer_edge_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
├── metadata/
└── dataset.yaml
```

## 5. 訓練

Segmentation：

```bash
python train.py \
  --data wafer_edge_dataset/dataset.yaml \
  --model yolov8m-seg.pt \
  --epochs 100 \
  --imgsz 512 \
  --batch 16
```

Detection：

1. 把 config.yaml 的 task 改成 `detect`
2. 重新執行 preprocess.py
3. 使用 detection 權重：

```bash
python train.py \
  --data wafer_edge_dataset/dataset.yaml \
  --model yolov8m.pt \
  --epochs 100 \
  --imgsz 512 \
  --batch 16
```

權重名稱可換成公司環境已下載的版本。

## 6. 單張 wafer 推論

```bash
python infer.py \
  --config config.yaml \
  --model runs/wafer_edge/train/weights/best.pt \
  --image sample.png \
  --conf 0.25 \
  --iou 0.5
```

輸出：

```text
inference_output/
├── sample_result.png
├── sample_result.json
└── sample_unwrapped.png
```

## 7. 每張 wafer 的圓心不同

建立 `wafer_geometry.json`：

```json
{
  "wafer_001": {
    "center_x": 2047.5,
    "center_y": 2051.0,
    "inner_radius": 1800,
    "outer_radius": 2025
  },
  "wafer_002.png": {
    "center_x": 2042.0,
    "center_y": 2048.0,
    "wafer_radius": 2000,
    "inner_margin": 200,
    "outer_margin": 25
  }
}
```

config.yaml：

```yaml
center_json: wafer_geometry.json
```

目前 `preprocess.py` 會使用每張圖的個別資料；
`infer.py` 範例則先使用 config 內固定值，若要批次推論，可套用同一個
`get_geometry()` 邏輯。

## 8. 重要注意事項

- train/val/test 必須先以 wafer ID 或 lot ID 分割，再產生 patch。
- 不可把同一顆 wafer 的不同 patch 隨機分到 train 與 val。
- val/test 不要抽正常 patch，才能正確評估 false positive。
- `unwrap_width` 太小會降低圓周方向解析度。
- `patch_width` 越小，缺陷相對越大，但上下文也越少。
- 建議先看 `*_unwrapped.png`，確認膠路完整且沒有切錯半徑。
- 推論的 NMS 是簡化版本；正式產線可改成展開座標上的 seam-aware 合併。
