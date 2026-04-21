# Main change for this branch
- Attention Figures add corresponding images + show more info
- add sample ordering logics to `dataset.py`
  
# Overview
## 測試
```
local_items, global_items = load_sample_meta(
    sample_dir=sample_dir,
    local_fov_threshold=LOCAL_FOV_THRESHOLD,
    max_local=MAX_LOCAL_VIEWS,
    max_global=MAX_GLOBAL_VIEWS
)

print(sample_name)
print("local:", [(x["image_name"], x["fov"]) for x in local_items])
print("global:", [(x["image_name"], x["fov"]) for x in global_items])
```

