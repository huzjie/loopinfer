# 推测解码详解

草稿模型每 token 快 `draft_speedup` 倍，连猜 K 个 token，主模型一次并行验证。

- 加速比 ≈ K / (1 + K / draft_speedup)
- 草稿 top-1 命中率越高，最优 K 越大
- 拒绝后按主模型 argmax 改判

```python
from loopinfer import InferenceEngine
engine = InferenceEngine()
accepted, draft = engine.speculate(1)
print(engine.spec_speedup())
```
