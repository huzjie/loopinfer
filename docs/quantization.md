# 量化详解

## 浮点格式

NVFP4 = 1 sign + 2 exp + 1 mant；MXFP8 = 1 sign + 4 exp + 3 mant。

## 块级 scale

每 `block_size` 个值共享一个 scale，反量化 = `code -> float -> * scale`。
NVFP4 常用 block_size=16，MXFP8 常用 32。

## 使用

```python
from loopinfer.quant import QuantConfig, quantize, dequantize
qc = QuantConfig(format="nvfp4", block_size=16)
codes, scales = quantize(weights, qc)
restored = dequantize(codes, scales, qc)
```
