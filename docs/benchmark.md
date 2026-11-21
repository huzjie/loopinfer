# 基准说明

参考规模（dim=32 / 32 专家 / 24 层 / 循环 6~18 ×3）在 CPU 上：

- `doctor`：10 项自检，秒级完成
- `generate` 64 token：毫秒~秒级
- `benchmark`：loop_passes 1/2/3/4 → eff_depth 24/36/48/60，n_params 恒 1736704

结论：**参数量不随深度增长**，深度（表达能力）随 passes 线性涨，算力随 passes 线性涨。
