import json
import os


def split_and_sort_sample_items(meta, local_fov_threshold):
    """
    將 meta["images"] 依照 threshold 分成 local / global，並排序：
      - local: fov 由大到小
      - global: fov 由小到大

    若 fov 相同，Python sort 會保留原始順序（穩定排序）。
    """
    local_items = []
    global_items = []

    for item in meta["images"]:
        file_name = item["file"]
        fov = item["fov"]

        info = {
            "image_name": file_name,
            "fov": fov
        }

        if fov <= local_fov_threshold:
            local_items.append(info)
        else:
            global_items.append(info)

    local_items.sort(key=lambda x: x["fov"], reverse=True)
    global_items.sort(key=lambda x: x["fov"])

    return local_items, global_items


def apply_branch_fallback(local_items, global_items):
    """
    若某個 branch 沒有圖，使用另一 branch 的第一張補上。
    與 dataset / infer 共用相同邏輯。
    """
    local_items = list(local_items)
    global_items = list(global_items)

    if len(local_items) == 0 and len(global_items) > 0:
        local_items.append({
            "image_name": global_items[0]["image_name"],
            "fov": global_items[0]["fov"]
        })

    if len(global_items) == 0 and len(local_items) > 0:
        global_items.append({
            "image_name": local_items[0]["image_name"],
            "fov": local_items[0]["fov"]
        })

    return local_items, global_items


def truncate_branch_items(local_items, global_items, max_local, max_global):
    """
    與 dataset padding 前的截斷邏輯一致。
    """
    local_items = local_items[:max_local]
    global_items = global_items[:max_global]
    return local_items, global_items


def load_and_prepare_sample_items(
    sample_dir,
    local_fov_threshold,
    max_local=None,
    max_global=None
):
    """
    從 sample_dir/meta.json 載入，並完成：
      1. split
      2. sort
      3. fallback
      4. truncate (optional)
    """
    meta_path = os.path.join(sample_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    local_items, global_items = split_and_sort_sample_items(
        meta=meta,
        local_fov_threshold=local_fov_threshold
    )

    local_items, global_items = apply_branch_fallback(local_items, global_items)

    if max_local is not None and max_global is not None:
        local_items, global_items = truncate_branch_items(
            local_items, global_items, max_local, max_global
        )

    return local_items, global_items
