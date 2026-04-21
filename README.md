# Main change for this branch
- Imageencoder to DinoV2
  
# Overview
## 改動
1. `configs/config.py`
2. `data/dataset.py`：把 transform 改成比較適合 ImageNet-pretrained backbone 的形式，加上 Normalize(...)。DINOv2 是 Vision Transformer backbone，先至少用標準 ImageNet normalization 比較合理。
3. `models/defect_model.py`：`dinov2_local` 用的是 
- 本地 `facebookresearch/dinov2` repo
- `torch.hub.load(..., source="local")`
- 再手動覆蓋你本地 `.pth`
4. `train.py`：這裡只要把 DINO_REPO_DIR 傳進 model
5. `infer.py`：同樣只是在 model 初始化時傳進 DINO_REPO_DIR。
## 測試
```
import torch

repo_dir = "./third_party/dinov2"
model = torch.hub.load(repo_dir, "dinov2_vitb14", source="local", pretrained=False)
print(type(model))
x = torch.randn(2, 3, 224, 224)
y = model(x)
print(y.shape)
```

