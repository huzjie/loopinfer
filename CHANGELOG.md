# Changelog

## 1.0.0 (2026-11-21)

- 首个版本：循环 Transformer（Looped MoE）推理运行时
- 循环层调度 + 跨 pass 消息传递（layer-level memory）
- top-k 稀疏 MoE 路由 + 负载均衡损失
- 推测解码草稿模型（draft + parallel verify）
- NVFP4 / MXFP8 浮点量化 + 反量化
- 确定性可训练 mock 后端（skill 参数单调趋 1）
- CLI 6 命令 / OpenAI 兼容 HTTP / Py+TS SDK / Docker/K8s/Helm/CI
