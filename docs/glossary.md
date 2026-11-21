# 术语表

| 术语 | 含义 |
|---|---|
| Looped Transformer | 中间若干层参数共享、前向循环执行多次的 Transformer |
| Recurrent depth | 循环层带来的「等效深度」 |
| Cross-pass message | 循环层之间传递的 layer-level 状态向量 |
| Equivalent depth | 前置 + passes×循环 + 后置 的等效层数 |
| Sparse MoE | 每 token 只激活 top-k 个专家的混合专家结构 |
| Load-balancing loss | 让专家被均匀使用的辅助损失，防止门控塌缩 |
| Speculative decoding | 小草稿模型连猜、主模型并行验证的加速解码 |
| NVFP4 / MXFP8 | 4-bit / 8-bit 浮点量化格式 |
