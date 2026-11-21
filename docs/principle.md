# 循环 Transformer 原理与等效深度推导

## 1. 问题：深度与参数的绑定

标准 Transformer 的「深度」就是层数 L，每层一组独立参数。想要更强的抽象能力，
就得加层、加参数，两者线性绑定。这对「本地部署」是致命的——参数量 = 显存 / 磁盘，
深度 = 算力，而笔记本两项都缺。

## 2. 循环层：解耦深度与参数

循环 Transformer 把中间若干层标记为「循环层」，参数只存一份，前向时循环执行 R 次：

```
等效深度 D = P + R * M + S
P = 前置层数（loop_start）
M = 循环层数（loop_end - loop_start）
S = 后置层数（n_layers - loop_end）
R = 循环次数（loop_passes）
```

参数只存 `P + M + S = L` 层，深度却达到 D（D >= L）。Inkling 2 Mini：L=24，M=12，R=3 →
D = 6 + 36 + 6 = 48。

## 3. 为什么循环不是「无脑重复」

同一层原样跑 3 次，第 2、3 次输入几乎不变，收益会衰减。因此引入 **cross-pass message**：
每个循环层维护一个 layer-level memory 向量，pass k 结束时把该层输出投影成 summary，
作为 pass k+1 的额外输入。于是每趟前向都「带上上一趟的结论」，形成逐趟递进，
类似 RNN 的隐藏状态，但作用在层粒度上。

## 4. 代价：算力换参数

深度翻倍、参数不变，代价是每 token 的算力（FLOPs）按等效深度走。所以循环层最适合
「显存紧、算力相对宽裕」的本地 / 端侧场景；若按 token 计费的云端，「纯堆层」更划算。
选型只回答一个问题：**你缺的是参数，还是缺的是算力？**

## 5. 本项目如何落地

`loop/scheduler.py` 实现三段调度，`loop/messaging.py` 实现 cross-pass memory。
`LoopConfig.eff_depth` 给出等效深度，供分析口径使用（注意：它不是算力口径）。
