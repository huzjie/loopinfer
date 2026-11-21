# API 参考

## Python

```python
from loopinfer import InferenceEngine

engine = InferenceEngine()                       # 默认参考规模
r = engine.generate("循环层的等效深度", max_tokens=32)
print(r.text)

print(engine.describe())                          # 配置 + 参数量 + 推测解码加速比
accepted, draft = engine.speculate(1)             # 一次推测解码
```

## 构造自定义模型

```python
from loopinfer import build_model, ModelConfig

cfg = ModelConfig(n_layers=24, loop_start=6, loop_end=18, loop_passes=3, n_experts=256, top_k=8)
model = build_model(cfg)
logits, hidden = model.forward_token(1)
```

## TypeScript SDK

见 `ts-sdk/`，含 `LoopInferClient`（HTTP 客户端）。

## HTTP

`POST /v1/completions`：`{"prompt": "...", "max_tokens": 32}` → OpenAI 兼容响应。
