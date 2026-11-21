# Benchmark

跑法：

```bash
python -m loopinfer benchmark --loop-passes 1 2 3 4
python scripts/benchmark.py
```

结果见 `results.example.md`。核心结论：参数量不随深度涨，深度随 passes 线性涨。
