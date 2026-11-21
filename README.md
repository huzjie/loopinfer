# LoopInfer — 循环 Transformer（Looped MoE）推理运行时

**把一个「26.4B 参数、激活仅 3.4B、中间 12 层循环跑 3 次等效 48 层深度」的循环 Transformer，**
**拆成四段可复用的工程：循环层参数共享、跨 pass 状态传递、MoE 稀疏路由、推测解码草稿模型。**

> 背景一句带过：Thinking Machines Lab（前 OpenAI CTO Mira Murati 创办）开源 Inkling 2 Mini——
> 一个「循环 Transformer + MoE」模型，总参 26.4B、每 token 激活 3.4B，24 层里中间 12 层
> 每次前向重复跑 3 次（等效 48 层级联），支持 1M token 上下文，检查点仅约 20GB，能在 32GB 内存的
> MacBook / DGX Spark / 32GB 显存 GPU 上本地跑。它凭什么用「小体积」换到「同规模 SOTA」？
> 核心就是 **循环层（looped layers）用时间换深度**——参数共享 + 跨 pass 消息传递。
> 本项目不是复刻 Inkling 权重，而是把这套思路落成一个**填配置即可跑**的纯 Python 推理运行时，
> 让你能照着手势把「循环深度」这门技术搬到自己项目里。

---

## 一句话原理

传统 Transformer 想更深、更强，只能堆更多层、堆更多参数。循环 Transformer 反其道：
**让中间若干层共享同一组参数，前向时把它们循环执行 N 次**，于是「L 层参数」能跑出
「远大于 L 的等效深度」，同时每个 pass 之间用一条 **cross-pass message** 传递状态，
避免循环变成纯重复、失去「逐层提炼」的能力。参数不变、深度翻倍，这就是 Inkling 2 Mini
用 3.4B 激活打平 Qwen3.8-27B 的根。

- **零依赖内核**：`dim=32 / 32 专家` 的参考规模在 CPU 上毫秒级跑通，不装 numpy/torch（官方 256 专家量级只需改 `n_experts`/`dim`，调度与解码逻辑不变）。
- **循环层调度**：`loop_start / loop_end / loop_passes` 三段配置，等效深度 = 前置 + passes×循环 + 后置。
- **跨 pass 消息**：每个循环层维护一个 layer-level memory 向量，pass k 的输出投影成 pass k+1 的额外输入。
- **MoE 稀疏路由**：top-k 专家门控 + 负载均衡损失，算力只随激活专家数走。
- **推测解码**：小草稿模型连猜 K 个 token，主模型一次前向并行验证，加速比 = K / (1 + K/draft_speedup)。
- **量化存储**：NVFP4 / MXFP8 两种浮点格式的对称量化 + 反量化，20GB 级检查点搬到笔记本。

---

## 目录

- [快速开始](#快速开始)
- [技术拆解：循环层怎么用小参数跑出等效深度](#技术拆解循环层怎么用小参数跑出等效深度)
- [干货块 A：循环层 + 跨 pass 消息伪代码（可直接照抄）](#干货块-a循环层--跨-pass-消息伪代码可直接照抄)
- [干货块 B：循环深度 vs 参数量 vs 算力权衡表](#干货块-b循环深度-vs-参数量-vs-算力权衡表)
- [干货块 C：MoE 稀疏路由 + 负载均衡实现要点](#干货块-cmoe-稀疏路由--负载均衡实现要点)
- [干货块 D：推测解码草稿模型工作流](#干货块-d推测解码草稿模型工作流)
- [干货块 E：NVFP4 / MXFP8 量化部署要点](#干货块-envfp4--mxfp8-量化部署要点)
- [踩过的坑](#踩过的坑)
- [项目结构](#项目结构)
- [CLI 参考](#cli-参考)
- [文档索引](#文档索引)

---

## 快速开始

```bash
# 1. 零依赖，直接跑自检（10 项）
python -m loopinfer doctor

# 2. 用参考规模跑通「循环 3 次」的推理链路
python -m loopinfer generate --prompt "循环 Transformer 的等效深度是怎么算出来的？" --max-tokens 64

# 3. 看循环深度 vs 参数量的权衡（等效深度从 24 到 96）
python -m loopinfer benchmark --loop-passes 1 2 3 4

# 4. 启动 OpenAI 兼容 HTTP 服务
python -m loopinfer serve --port 8080

# 5. 换真实权重：只替换 model 层的数值即可，其余调度 / MoE / 解码逻辑全部复用
```

填一份 JSON 配置（见 `config/loopinfer.example.json`）就能把 `loop_start/loop_end/loop_passes`、
`n_experts/top_k`、`drafter`、`quant` 全改掉，不用动代码。

---

## 技术拆解：循环层怎么用小参数跑出等效深度

### 1. 从「堆层」到「循环层」

普通 Transformer 深度 = 层数 L，参数随 L 线性涨。循环 Transformer 把网络切成三段：

```
[前置层 0..loop_start)     —— 一次性执行，参数不共享
[循环层 loop_start..loop_end) —— 参数共享，循环执行 loop_passes 次
[后置层 loop_end..L)       —— 一次性执行，参数不共享

等效深度 = loop_start + loop_passes × (loop_end - loop_start) + (L - loop_end)
```

Inkling 2 Mini 参考：L=24，中间 12 层（6..18）循环 3 次 → 6 + 3×12 + 6 = **48 层等效深度**，
但参数量只有 24 层。**深度（表达能力）涨了，参数（内存/存储）没涨。**

### 2. 跨 pass 消息：让循环不是「无脑重复」

如果只是把同一层原样跑 N 次，第 2、3 次的输入和前一次几乎一样，循环收益会边际递减。
解法是给每个循环层挂一个 **layer-level memory 向量**：

- pass k 时，层输入 = token 隐状态 + gate × memory[layer]（上一 pass 该层状态的投影）；
- pass 结束时，用该 pass 该层输出的一个 summary 更新 memory[layer]；
- 于是每个 pass 看到的是「输入 + 上一次的提炼结果」，形成类似 RNN 的逐 pass 递进，而不是重复。

这正是「循环机制让模型思考更多、输出更少」的工程来源——**同参数，多趟前向，每趟带上上趟的结论**。

### 3. 为什么激活参数这么低（MoE）

总参 26.4B，每 token 只激活 3.4B：因为每层是 MoE，256 个路由专家 + 1 共享专家，
门控只挑 top-8 路由专家计算。**存储 26.4B、每次只算 3.4B**，内存大、算力小，天生适合本地部署。

### 4. 为什么检查点只有 20GB（量化）

NVFP4（4-bit）与 MXFP8（8-bit）两种浮点量化把 26.4B 权重压到 ~20GB，反量化后再进计算图。
FP8 保精度、FP4 保体积，按层敏感度混用。

---

## 干货块 A：循环层 + 跨 pass 消息伪代码（可直接照抄）

```python
# 循环 Transformer 前向：参数共享 + 跨 pass 状态传递
def forward_layers(x, layers, loop_start, loop_end, loop_passes, mem_proj, gate):
    # 1) 前置层，一次性
    for idx in range(loop_start):
        x = layers[idx](x)

    # 2) 循环层：参数共享，重复 loop_passes 次
    memory = [None] * (loop_end - loop_start)  # 每个循环层一个 layer-level 状态
    for p in range(loop_passes):
        for j, idx in enumerate(range(loop_start, loop_end)):
            # 跨 pass 消息：把上一 pass 的层状态注入当前输入
            carry = memory[j]
            if carry is not None:
                x = x + gate * mem_proj[j](carry)
            # 正常执行这一层，同时输出一个 summary 作为下一 pass 的消息
            x, summary = layers[idx](x, return_summary=True)
            memory[j] = summary

    # 3) 后置层，一次性
    for idx in range(loop_end, len(layers)):
        x = layers[idx](x)

    return x

# 等效深度（只用于描述/分析，不影响前向）
eff_depth = loop_start + loop_passes * (loop_end - loop_start) + (len(layers) - loop_end)
```

**照抄要点**：`memory` 是 layer-level 的（每个循环层一条），不是 token-level；`gate` 用可学习标量
（初始 0，训练中逐渐打开），避免首 pass 就被不稳定的 memory 带偏。这套结构把「深度」从
「多少层」解耦成「多少 pass」，是循环 Transformer 最值得搬走的一块。

---

## 干货块 B：循环深度 vs 参数量 vs 算力权衡表

| 方案 | 层数 L | 循环区间 | passes | 等效深度 | 参数量 | 每 token 算力 | 适用 |
|---|---|---|---|---|---|---|---|
| 纯堆层 | 48 | 无 | 1 | 48 | 48× | 48× | 服务器 / 云端 |
| 轻度循环 | 24 | 12 层 | 2 | 36 | 24× | 36× | 显存紧但算力足 |
| **Inkling 式** | 24 | 12 层 | 3 | 48 | 24× | 48× | 笔记本本地 |
| 深度循环 | 12 | 8 层 | 6 | 54 | 12× | 54× | 极致小参、延迟敏感 |

**可复用结论**：循环层用「算力换参数」——参数量砍半，深度不减，但每 token 算力（FLOPs）按等效深度走。
所以它最适合**「内存/显存是瓶颈、算力不是瓶颈」**的场景：本地部署、端侧、边缘。若你反过来
「算力贵、内存便宜」（云端按 token 计费），纯堆层更划算。选型前先回答一个问题：
**你缺的是参数（显存）还是缺的是深度（算力）？**

---

## 干货块 C：MoE 稀疏路由 + 负载均衡实现要点

```python
# top-k 稀疏门控：logits -> 选 top-k 专家 -> softmax 归一 -> 加权求和
def route(x, gate_w, n_experts, top_k, experts, load_bal_coef):
    logits = x @ gate_w                      # [T, n_experts]
    topk_vals, topk_idx = topk(logits, top_k)
    weights = softmax(topk_vals)             # 只在 top-k 内归一
    out = sum(weights[k] * experts[topk_idx[k]](x) for k in range(top_k))

    # 负载均衡损失：让每个专家被选中的概率尽量均匀，避免"赢家通吃"塌缩
    frac = one_hot(topk_idx).mean(0)         # 每个专家被选中占比
    load_bal_loss = load_bal_coef * n_experts * sum(frac * (1.0 / n_experts - frac))
    return out, load_bal_loss
```

**三个坑**：① softmax 一定要**只在 top-k 内归一**，若对全部 logits 归一，被淘汰专家的权重没归零，
稀疏就名存实亡；② 共享专家（shared expert）不走门控，恒参与，保证最差情况也有一个专家兜底；
③ **不加载负载均衡损失，门控会塌缩到 1~2 个专家**，稀疏度是上去了，但剩下专家的参数等于白存，
且那几个专家过载——load_bal_loss 是稀疏 MoE 能训起来的关键约束，不是可选项。

---

## 干货块 D：推测解码草稿模型工作流

```python
# 草稿模型连猜 K 个 token，主模型一次前向并行验证
def speculate(prefix, drafter, model, K):
    draft_tokens = []
    x = prefix
    for _ in range(K):                       # 草稿快、逐 token 连猜
        t = drafter.sample(x)
        draft_tokens.append(t)
        x = x + [t]
    # 主模型一次前向，拿到 K+1 个位置的概率分布
    probs = model.forward(prefix + draft_tokens)   # 并行！不是逐 token
    accept = 0
    for i in range(K):
        if draft_tokens[i] == argmax(probs[i]):    # 验证通过就接受
            accept += 1
        else:
            draft_tokens[i] = sample(probs[i])     # 第一次拒绝就改判并停
            break
    return prefix + draft_tokens[:accept + 1]
```

**可复用要点**：草稿模型每 token 快 `draft_speedup` 倍，主模型一次并行算 K 个位置，理论上
加速比 ≈ `K / (1 + K/draft_speedup)`。草稿越准、K 越大，收益越高，但 K 过大会让「拒绝后浪费的
草稿算力」吃掉收益。工程上把 K 做成可调，按草稿 top-1 命中率动态选 K——命中率高就多猜几个。

---

## 干货块 E：NVFP4 / MXFP8 量化部署要点

| 格式 | 位数 | 指数 | 尾数 | 用途 | 典型粒度 |
|---|---|---|---|---|---|
| NVFP4 | 4-bit | 2 | 1 | 存储压缩到极致（20GB 级） | 每 16 个值共享 scale |
| MXFP8 | 8-bit | 4 | 3 | 精度优先的关键层 | 每 32 个值共享 scale |

**要点**：① 浮点量化（非整型）用「共享 scale + 逐元素小尾数」，反量化就是 `x = scale * (1 + m/2^mbits) * 2^e`，
比 INT8 对「接近 0 的梯度/激活」更友好；② **按层敏感度混用**——embedding / 输出头 / 门控用 FP8，
MoE 专家 FFN 的稠密矩阵用 FP4（体积大头）；③ 量化只在「存储 / 传输」层做，**反量化回 FP32 再进计算图**，
推理精度损失可控；④ 关键：FP4 的 scale 一定要**按 block 共享**（不是全局一个 scale），否则异常值把
整层 scale 拉爆。

---

## 踩过的坑

1. **循环层别无脑重复**——不加 cross-pass 消息时，pass 2/3 的输出和 pass 1 高度相关，循环收益边际递减；
   必须给每层挂 memory 状态（干货块 A），循环才有「逐趟提炼」的价值。
2. **等效深度是「描述口径」不是「算力口径」**——循环 3 次的算力就是 3 倍单层，别拿等效深度去估 FLOPs。
3. **top-k 内 softmax**——对全量 logits 归一会让稀疏失效（干货块 C 坑 ①）。
4. **负载均衡损失不能省**——否则 MoE 塌缩（干货块 C 坑 ③）。
5. **FP4 scale 必须分块**——全局单 scale 会被异常值拉爆（干货块 E 坑 ④）。
6. **草稿 K 不是越大越好**——K 太大时拒绝浪费的草稿算力反超收益（干货块 D）。
7. 从子目录跑示例 / 测试时，先 `pip install -e .` 或设 `PYTHONPATH=.`，否则找不到 `loopinfer` 包。

---

## 项目结构

```
loopinfer/
├── loopinfer/            # 纯 Python 零依赖内核
│   ├── core/             # 状态 / 采样器 / 注册表 / 指标 / 错误 / 事件
│   ├── loop/             # ★ 循环层调度 + 跨 pass 消息（本项目的核心）
│   ├── moe/              # top-k 稀疏门控 + 专家 FFN + 负载均衡
│   ├── attention/        # 多头注意力 + 1M 上下文滑窗
│   ├── model/            # Transformer 主干 / rotary / layernorm / embedding
│   ├── drafter/          # 推测解码草稿模型 + 验证
│   ├── tokenizer/        # 确定性 BPE 分词器
│   ├── quant/            # NVFP4 / MXFP8 量化 + 反量化
│   ├── engine/           # 推理 / prefill / 批处理 / 流式 / session
│   └── cli.py            # doctor / generate / benchmark / serve / config / version
├── config/               # 示例 JSON 配置（循环 / MoE / drafter / quant 全可调）
├── prompts/              # 可复用的提示词模板（含本项目的工程提示词）
├── examples/             # quickstart / 循环深度对比 / 推测解码演示
├── tests/                # 单元测试（44+）
├── docs/                 # 原理 / 调参 / 部署 / 迁移 多篇
├── ts-sdk/               # TypeScript SDK
├── deploy/               # Docker / K8s / Helm / CI
└── notebooks/            # 循环深度可视化
```

---

## CLI 参考

| 命令 | 作用 |
|---|---|
| `python -m loopinfer doctor` | 10 项自检（config/tokenizer/loop/moe/attention/model/drafter/quant/generate/registry） |
| `python -m loopinfer generate` | 用参考规模跑通完整推理链路 |
| `python -m loopinfer benchmark --loop-passes 1 2 3 4` | 对比不同循环深度 |
| `python -m loopinfer serve --port 8080` | OpenAI 兼容 HTTP 服务 |
| `python -m loopinfer config` | 解析并 dump 当前配置 |
| `python -m loopinfer version` | 版本信息 |

---

## 文档索引

- `docs/principle.md` — 循环 Transformer 原理与等效深度推导
- `docs/tuning.md` — loop_passes / top_k / drafter K 的调参指南
- `docs/deploy.md` — Docker / K8s / Helm 部署
- `docs/migration.md` — 从纯堆层 Transformer 迁移到循环层的步骤
- `docs/api.md` — Python / TypeScript SDK API

---

## License

Apache-2.0。核心思路参考 Thinking Machines Lab Inkling 2 Mini 的公开论文 / 技术说明，
本仓库为独立、零依赖的**教学与工程参考实现**，不含任何官方权重。
