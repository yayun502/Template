# Main change for this branch
- Branch gate：branch-level attention，在 local feature 和 global feature 之間決定權重
  
# Overview
## 架構
```
local images  ── encoder ── local attention pooling  ── local_feat
                                                          │
                                                          ├── branch gate ── local_weight
                                                          │
global images ─ encoder ── global attention pooling ── global_feat
                                                          │
                                                          └── branch gate ── global_weight

weighted_local = local_weight * local_feat
weighted_global = global_weight * global_feat

concat(weighted_local, weighted_global)
        │
fusion
        │
main head + hierarchical head
```

