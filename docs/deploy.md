# 部署指南

## Docker

```bash
docker build -t loopinfer .
docker run --rm -p 8080:8080 loopinfer serve --port 8080
```

## Kubernetes / Helm

```bash
kubectl apply -f deploy/k8s/deployment.yaml
# 或
helm install loopinfer deploy/helm/loopinfer
```

## 本地

```bash
python -m loopinfer doctor        # 自检
python -m loopinfer serve --port 8080
curl http://127.0.0.1:8080/v1/completions -d '{"prompt":"hello"}'
```

## 换真实权重

替换 `loopinfer/model/` 下的数值层即可，调度 / MoE / 解码逻辑全部复用。量化走
`loopinfer.quant`：`quantize()` 把 FP32 权重压成 codes + scales，`dequantize()` 还原。
