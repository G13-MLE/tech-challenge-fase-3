# Tech Challenge Fase 3: Sistema de Triagem de Laudos Médicos

Sistema de triagem automática de exames de texto (laudos médicos) para classificação de urgência (normal / atenção / urgente), servido via API REST em container Docker.

## Decisões Arquiteturais: Etapa 1

### Estratégia de deploy: Real-time (AWS ECS/Fargate)

**Cenário**: triagem clínica exige resposta imediata, o laudo não pode esperar um job agendado.

| Estratégia | Latência | Custo | Complexidade | Adequação |
|---|---|---|---|---|
| Batch | Alta, tolerada | Menor | Baixa | Não, triagem não espera job |
| Real-time | Baixa, determinística | Maior (infra sempre ativa) | Alta | **Escolhida** |
| Serverless | Variável (cold start) | Eficiente em cargas intermitentes | Média | Alternativa citada, cold start impacta latência |

**Escolha**: AWS ECS/Fargate com container Docker.
- Modelo leve (TF-IDF + RandomForest) carregado em memória no startup da API
- Escalabilidade horizontal via ECS, sem gerenciamento de servidores (Fargate)
- Latência previsível e baixa, requisito crítico para triagem

### Tecnologia

| Componente | Decisão |
|---|---|
| API | FastAPI (`/health`, `/predict`) |
| Modelo | Scikit-Learn (TF-IDF + RandomForest/LogisticRegression), artefato em `joblib` + `onnx` |
| Container | Docker multi-stage, `python:3.14-slim` |
| Benchmark | Percentis P50/P95/P99 (nunca média), comparativo joblib vs ONNX |

### Segurança e FinOps

- Rate limiting para proteção contra abuso
- Menor privilégio (IAM, acesso restrito ao modelo)
- Proteção contra Model Extraction (limitar exposição do modelo)

### Mapeamento de Urgência

O modelo classifica o texto em 5 classes clínicas, mapeadas para 3 níveis de urgência:

| Classe clínica | Urgência |
|---|---|
| Pneumonia | Urgente |
| Diabetes / Hipertensão | Atenção |
| Asma / Hérnia | Normal |

> Nota: na Etapa 1 a API usava um modelo dummy em `joblib`. Na Etapa 4, o modelo real é treinado com o Medical Abstracts TC Corpus (5 classes clínicas → 3 níveis de urgência), exportado para ONNX para inferência otimizada. O dataset Kaggle é baixado via `make data-kaggle`.

## Início Rápido

### Requisitos

- Python 3.14+
- uv (gerenciador de pacotes)
- Docker (opcional, para container)

### Instalação

```bash
make setup
```

### Gerar modelo via pipeline

```bash
make data-synthetic   # dataset sintético registrado no DVC
make data-kaggle      # dataset Kaggle (Medical Abstracts TC) registrado no DVC
make pipeline         # dvc repro: ingestão -> treino -> avaliação -> onnx
```

### Remote DVC (opcional)

O pipeline é **self-contained** por padrão: dados e modelos ficam no cache
local do DVC, sem necessidade de remote. Para compartilhar artefatos entre
máquinas (ex.: via OneDrive), preencha `DVC_ONEDRIVE_REMOTE_URL` no `.env`
e configure o remote:

```bash
make dvc-remote        # ou make setup (configura automaticamente)
```

Para remover o remote: `dvc remote remove onedrive`.

### Executar localmente

```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Executar em Docker

```bash
make docker-run
```

## Airflow (perfil airflow)

Stack mínima containerizada: Airflow 3.x com `LocalExecutor` (PostgreSQL
para metadados, sem Redis/Celery) + MLflow para tracking de experimentos.
Containers: `airflow-postgres`, `airflow-init`, `airflow-scheduler`,
`airflow-dag-processor`, `airflow-apiserver` (UI + API) e `mlflow`.

### Subir a stack

```bash
make airflow-up
```

Aguardar os containers ficarem healthy (~30s). Acessar:

| Serviço | URL |
|---|---|
| Airflow UI | http://localhost:8080 (admin/admin) |
| MLflow UI | http://localhost:5001 |

### Parar a stack

```bash
make airflow-down
```

A DAG `train_pipeline` executa semanalmente (`@weekly`) e reutiliza as funções de treino do pipeline DVC via `src/orchestration/training_tasks`.

## Monitoramento (perfil monitoring)

Stack Prometheus + Grafana containerizada. A API expõe `/metrics` com contadores e histogramas Prometheus; o Grafana provisiona automaticamente datasource e dashboard.

### Subir a stack

```bash
make monitoring-up
make generate-traffic
```

Aguardar os containers ficarem healthy. Acessar:

| Serviço | URL |
|---|---|
| API | http://localhost:8000 |
| Métricas | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / `$GRAFANA_ADMIN_PASSWORD`) |

O dashboard "Triage API" é provisionado automaticamente com 3 painéis: taxa de requisições, latência P95 e taxa de erro (5xx).

### Parar a stack

```bash
make monitoring-down
```

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Health check (ok / degraded) |
| GET | `/metrics` | Métricas Prometheus |
| POST | `/api/v1/predict` | Classificação de urgência |
| POST | `/api/v1/predict/batch` | Classificação em lote (até 100 textos) |

Exemplo:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "paciente com pneumonia bacteriana grave"}'
```

Autenticação opcional por API key (`X-API-Key`): habilite via `API_KEY_ENABLED=true` e defina `API_KEY` no `.env`.

### Benchmark de latência

Benchmark com um único backend (atual):

```bash
make benchmark
```

Benchmark comparativo joblib vs ONNX (reinicia a API com cada backend):

```bash
make benchmark-comparison
```

## Baseline de Latência

Resultados obtidos com `make benchmark` em Docker (modelo sintético, 100 requisições):

| Percentil | Latência |
|---|---|
| P50 | 9.61 ms |
| P95 | 15.39 ms |
| P99 | 31.35 ms |

## Comparação de Latência: joblib vs ONNX

> Preenchido após executar `make benchmark-comparison` com o modelo real.
> O relatório é salvo em `reports/benchmark_comparison.md`.

| Métrica | joblib | ONNX | Δ |
|---|---|---|---|
| P50 | — ms | — ms | — |
| P95 | — ms | — ms | — |
| P99 | — ms | — ms | — |
| Artefato | — KB | — KB | — |

## Comandos Disponíveis

```bash
make help
```

| Comando | Descrição |
|---|---|
| `make setup` | Configurar ambiente |
| `make test` | Rodar testes |
| `make lint` | Verificar código |
| `make data-synthetic` | Gerar dataset sintético e registrar no DVC |
| `make data-kaggle` | Baixar dataset Kaggle (Medical Abstracts TC) e registrar no DVC |
| `make pipeline` | Rodar pipeline DVC completo (dvc repro) |
| `make train` | Rodar estágio de treino (dvc repro train) |
| `make pipeline-live` | Rodar pipeline direto (sem DVC), logs live |
| `make train-live` | Rodar treino direto (sem DVC), logs live |
| `make evaluate-live` | Rodar avaliação direto (sem DVC), logs live |
| `make dvc-remote` | Configurar remote DVC OneDrive (opcional) |
| `make benchmark` | Benchmark de latência |
| `make benchmark-comparison` | Benchmark comparativo joblib vs ONNX |
| `make docker-build` | Construir imagem Docker |
| `make docker-run` | Rodar API em container |
| `make docker-train` | Pipeline DVC no container (perfil train) |
| `make airflow-up` | Subir stack Airflow + MLflow |
| `make airflow-down` | Parar stack Airflow |
| `make monitoring-up` | Subir API + Prometheus + Grafana |
| `make monitoring-down` | Parar stack de monitoramento |
| `make generate-traffic` | Gerar tráfego para popular métricas |
