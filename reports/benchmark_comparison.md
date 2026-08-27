## Comparação de Latência: joblib vs ONNX

Backend: **100 requisições** cada | Warmup: 10 req

| Métrica | joblib | ONNX | Δ |
|---------|-------|------|---|
| P50 | 10.77 ms | 9.13 ms | -1.64 ms (-15.2%) |
| P95 | 12.34 ms | 13.02 ms | +0.68 ms (+5.5%) |
| P99 | 46.97 ms | 18.58 ms | -28.39 ms (-60.4%) |
| min | 9.58 ms | 6.32 ms | -3.26 ms (-34.0%) |
| max | 47.24 ms | 18.59 ms | -28.66 ms (-60.7%) |

| Artefato | 37.44 MB | 21.96 MB | — |
