# 配置说明

`loopinfer.example.json` / `loopinfer.yaml` 是「填配置即可跑」的入口。三块核心：

- `model`：主干结构（循环区间 loop_start/loop_end、循环次数 loop_passes、MoE 专家数 n_experts/top_k）
- `drafter`：推测解码草稿模型（K 草稿长度、draft_speedup 草稿加速比）
- `quant`：量化格式（nvfp4 / mxfp8 + block_size）

用 `python -m loopinfer config --config config/loopinfer.yaml` 查看解析后的完整配置。
