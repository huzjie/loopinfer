# 架构决策

## 分层

```
engine (编排) -> model (主干) -> [loop 调度 + cross-pass memory]
                               -> [attention + moe]
                               -> [drafter + quant]
```

- `loop/`：三段调度 + 跨 pass 消息，是循环 Transformer 的核心，与具体层解耦。
- `moe/`：top-k 稀疏路由，专家**惰性生成**（避免 n_experts × n_layers 预生成拖慢 init）。
- `drafter/`：推测解码，草稿模型独立于主模型，可替换。
- `quant/`：NVFP4/MXFP8 量化，块级共享 scale，量化只在存储层做。

## 关键决策

1. **专家惰性生成**：MoELayer 只物化被 top-k 路由到的专家，避免 init 爆炸。
2. **token 级前向 + KV cache**：自回归推理架构，滑窗 KV cache 支持长上下文。
3. **cross-pass memory 是 layer-level**：每条循环层一个状态向量，非 token-level。
4. **参考规模 vs 官方值分离**：默认 dim=32/32 专家可在 CPU 跑；官方 26.4B/256 专家只改配置。
