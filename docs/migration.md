# 从纯堆层 Transformer 迁移到循环层

## 步骤

1. **选定循环区间**：通常把中间 50% 层设为循环层（loop_start = L*0.25, loop_end = L*0.75）。
2. **引入 cross-pass memory**：给每个循环层挂一个 layer-level memory 向量 + 可学习 gate（初始 0）。
3. **前向改成三段调度**：前置 → (循环层 × passes) → 后置，见 README 干货块 A。
4. **校准等效深度**：用 `eff_depth` 作为分析口径，但训练 / 推理的算力预算按「实际执行层数」算。
5. **验证收益**：同参数下，passes 2~3 应带来可测的 perplexity / 精度提升，否则说明循环区间或
   cross-pass 消息没调好（最可能是 memory 的 gate 没打开、或 summary 投影太弱）。

## 常见翻车点

- cross-pass gate 初始值设太大，首 pass 就被不稳定 memory 带偏 → gate 初始 0。
- 只循环不传消息，收益趋近于零 → 必须加 memory。
- 拿等效深度去估算力 → 会严重低估，循环 3 次就是 3 倍单层算力。
