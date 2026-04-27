# Main change for this branch
- Conditional gate loss
  
# Overview
## 概念
如果只是普通 gate loss：
```
is_np
is_single
has_breakpoint
```
確實會有點像把原本 4-class label 展開成 3 個 binary label。
但它還是有一點價值，因為它會迫使模型學到比較語意化的中間判斷。

不過你擔心的是對的：普通 gate loss 不夠像真正 flow chart。

所以我更建議用 conditional gate loss：
```
NP gate：所有樣本都算
Single gate：只在 non-NP 樣本上算
Breakpoint gate：只在 non-NP 且 non-Single 樣本上算
```
這樣它就不是單純 label extension，而是比較真的在模擬人類判斷順序。

