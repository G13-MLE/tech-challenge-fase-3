## Comparação de Latência: joblib vs ONNX

Backend: **100 requisições** cada | Warmup: 10 req

| Métrica | joblib | ONNX | Δ |
|---------|-------|------|---|
| P50 | 11.03 ms | 7.79 ms | -3.24 ms (-29.4%) |
| P95 | 17.42 ms | 10.53 ms | -6.89 ms (-39.6%) |
| P99 | 20.76 ms | 18.74 ms | -2.02 ms (-9.7%) |
| min | 9.72 ms | 5.48 ms | -4.24 ms (-43.6%) |
| max | 20.77 ms | 18.75 ms | -2.02 ms (-9.7%) |

| Artefato | 257.1 KB | 115.8 KB | — |
