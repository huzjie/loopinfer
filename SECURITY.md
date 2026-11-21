# Security Policy

## Reporting a Vulnerability

请勿在公开 Issue 中披露漏洞。发送邮件至 maintainer（见仓库联系方式）并附复现步骤，
我们将在 48 小时内响应。

## 注意事项

本仓库为教学与工程参考实现，不包含任何官方模型权重。部署到生产前请自行完成安全评估，
尤其注意：HTTP serve 模式默认绑定 localhost，如需对外暴露请置于反向代理之后。
