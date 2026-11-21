# Contributing

## 开发流程

1. Fork 本仓库，创建 feature 分支
2. 本地自检：`python -m loopinfer doctor`
3. 跑测试：`python -m pytest -q`
4. 提交 PR，说明改动与动机

## 代码风格

- 纯 Python、零第三方依赖
- 命名清晰、注释只在关键逻辑处
- 核心数值逻辑保持确定性（可复现）

## 新功能建议

开 Issue 描述场景与预期，或直接在 Discussion 投票。欢迎补齐：更多循环调度策略、
更细粒度的量化格式、多后端接入。
