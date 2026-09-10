## Comparação de Latência: joblib vs ONNX

Backend: **100 requisições** cada | Warmup: 10 req

| Métrica | joblib | ONNX | Δ |
|---------|-------|------|---|
| P50 | 12.03 ms | 8.80 ms | -3.23 ms (-26.8%) |
| P95 | 13.82 ms | 13.02 ms | -0.80 ms (-5.8%) |
| P99 | 45.89 ms | 21.35 ms | -24.54 ms (-53.5%) |
| min | 10.33 ms | 6.93 ms | -3.40 ms (-32.9%) |
| max | 46.12 ms | 21.37 ms | -24.76 ms (-53.7%) |

| Artefato | 37.44 MB | 115.8 KB | — |
