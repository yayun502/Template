# Main change for this branch
- class weight
- focal loss
- scheduler
- attention可視化
  
# Overview
## Class Weight
### 為什麼要加
工業資料常見類別不平衡。
如果某些類別樣本比較少，模型容易偏向大量類別，導致：
- 少數類 recall 偏低
- confusion matrix 主要集中在大類
- overall accuracy 看起來不差，但小類學不好
### 目的
在 cross entropy 或 focal loss 中，讓少數類的錯誤被懲罰得更重，減少模型偏向大類的現象。
### 效果預期
加入 class weight 後，常見現象是：
- 少數類的 recall 提升
- 某些混淆情況改善
- overall accuracy 不一定立刻上升，但 per-class 表現通常更平衡
### 相關 config
在 `configs/config.py`：
```
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS = [1.0, 2.0, 2.0, 1.5]
```
順序必須和 `LABEL_MAP` 一致：
- 0: `Single`
- 1: `NP`
- 2: `定點`
- 3: `Multi`
### 建議怎麼設
一開始先用保守值，例如：
```
CLASS_WEIGHTS = [1.0, 1.5, 1.5, 1.2]
```
如果某類明顯學不起來，再逐步增加權重。
### 推薦實驗組合
#### 不使用 class weight
```
USE_CLASS_WEIGHTS = False
```
#### 使用 class weight
```
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS = [1.0, 1.5, 1.5, 1.2]
```

## Focal Loss
### 為什麼要加
一般 cross entropy 會同等看待所有樣本，但訓練後期常常大量 easy samples 已經很容易，loss 仍被這些樣本主導。
對於：
- 難分樣本
- 容易混淆的類別
- 少數類樣本
一般 CE 可能不夠聚焦。
### 目的
Focal loss 會降低 easy samples 的影響，讓模型更專注在難分樣本上。
### 效果預期
常見改善方向：
- 難分類別 recall 提升
- 混淆類別的 decision boundary 更清楚
- 少數類與 hard examples 更容易被關注
但也要注意：
- train loss 不一定比 CE 低
- overall accuracy 不一定馬上更好
- 如果資料本身不太不平衡，未必比 CE 好
### 相關 config
```
CLS_LOSS_TYPE = "focal"
FOCAL_GAMMA = 2.0
```
### `gamma` 的意義
- `gamma = 0`：接近一般 CE
- `gamma = 1`：有些 focal 效果
- `gamma = 2`：常見預設，建議先從這裡開始
- `gamma > 3`：效果可能太強，訓練較不穩
### 推薦實驗組合
#### CE baseline
```
CLS_LOSS_TYPE = "ce"
USE_CLASS_WEIGHTS = False
```
#### CE + class weight
```
CLS_LOSS_TYPE = "ce"
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS = [1.0, 1.5, 1.5, 1.2]
```
#### Focal only
```
CLS_LOSS_TYPE = "focal"
FOCAL_GAMMA = 2.0
USE_CLASS_WEIGHTS = False
```
#### Focal + class weight
```
CLS_LOSS_TYPE = "focal"
FOCAL_GAMMA = 2.0
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS = [1.0, 1.5, 1.5, 1.2]
```
### 建議怎麼比較
不要只看 overall accuracy，應同時比較：
- confusion matrix
- per-class precision / recall / f1-score
- 特別關注 NP、定點、Multi 這些較容易不平衡或混淆的類別


## Scheduler
### 為什麼要加
如果 learning rate 固定，模型可能：
- 前期學得動
- 後期卡住卻沒有更細的更新步伐
scheduler 的作用是隨訓練過程調整 learning rate，讓優化更穩定。
### 目的
改善訓練後期收斂品質，避免固定 LR 導致停滯。
### 目前支援的 scheduler
#### 1. `none`
不使用 scheduler，learning rate 固定。
```
SCHEDULER_TYPE = "none"
```
#### 2. `step`
每隔固定 epoch 直接降低 learning rate。
```
SCHEDULER_TYPE = "step"
STEP_SIZE = 10
STEP_GAMMA = 0.1
```
#### 3. `cosine`
learning rate 平滑下降。
```
SCHEDULER_TYPE = "cosine"
COSINE_T_MAX = EPOCHS
COSINE_ETA_MIN = 1e-6
```
#### 4. `plateau`
當 validation 指標停滯時才下降。
```
SCHEDULER_TYPE = "plateau"
PLATEAU_MODE = "max"
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 3
```
### 效果預期
- 後期訓練較穩
- validation 指標不容易提早卡死
- 有時對 confusion matrix 的改善比對 overall acc 更明顯
### 建議先從哪個開始
最推薦先試：
```
SCHEDULER_TYPE = "cosine"
```
如果你比較在意 validation accuracy 卡住時再降 learning rate，可以試：
```
SCHEDULER_TYPE = "plateau"
```

## Attention 可視化
### 為什麼要加
這個任務不是單張圖分類，而是同一個 sample 由多張子圖共同決定。
因此很值得知道模型最依賴哪幾張子圖。
### 目的
檢查模型在 sample-level 多圖決策時，究竟比較重視：
- 哪些 low-FOV 子圖
- 哪些 high-FOV 子圖
- 是否與人類判斷一致
### attention 的意義
目前 attention 不是 pixel-level heatmap，而是：
- **local branch 內各張圖的重要性權重**
- **global branch 內各張圖的重要性權重**  
也就是在同一個 sample 的多張圖之間，模型在 local 或 global 分支中各自分配注意力。
### 目前輸出
#### 1. Attention CSV
`inference_outputs/test_attention_weights.csv`  
欄位包括：
- sample_name
- gt_cls
- pred_cls
- branch
- image_name
- fov
- attention_weight
#### 2. 每個 sample 的 attention 圖
`inference_outputs/attention_figures/*.png`  
每張圖會分成：
- 上半部：local branch attention
- 下半部：global branch attention
#### 如何解讀
##### 正確做法
比較 local branch 內部 哪張圖權重高
比較 global branch 內部 哪張圖權重高
##### 不建議直接比較
不要直接把 local 的 0.7 和 global 的 0.4 當成絕對重要性比較，因為兩個 branch 的 attention 是各自獨立 softmax。
##### 建議先看哪些 sample
正確且高信心的 sample
高信心但錯誤的 sample
定點 / Multi 混淆樣本
NP 類別樣本，看 low-FOV 是否真的被高權重選中
