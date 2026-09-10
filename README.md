# Tech Challenge Fase 3 — Sistema de Triagem de Laudos Médicos

Sistema de triagem automática de laudos médicos (NLP) para classificação de urgência
(normal / atenção / urgente), com pipeline reprodutível via **DVC**, modelo **TF-IDF +
LogisticRegression** com **ajuste de limiares** otimizado para **ONNX**, servido via
**FastAPI** em container Docker, com **CI/CD** (GitHub Actions), **orquestração**
(Airflow) e **monitoramento** (Prometheus + Grafana).

> Projeto do grupo **G13-MLE** para o Tech Challenge da Fase 03 (PÓS TECH FIAP).
> Dataset: **Medical Abstracts TC Corpus** (14.438 laudos, 5 classes clínicas → 3 níveis de urgência).

---

## Sumário

- [Visão geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Entregáveis por etapa](#entregáveis-por-etapa)
- [Critérios de avaliação](#critérios-de-avaliação)
- [Bibliotecas requeridas](#bibliotecas-requeridas)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Setup rápido](#setup-rápido)
- [Como executar o pipeline](#como-executar-o-pipeline)
- [Decisão arquitetural](#decisão-arquitetural)
- [Otimização de latência](#otimização-de-latência)
- [FinOps — Estimativa de custos AWS](#finops--estimativa-de-custos-aws)
- [Segurança](#segurança)
- [CI/CD](#cicd)
- [Histórico de commits](#histórico-de-commits)
- [Airflow](#airflow)
- [Monitoramento](#monitoramento)
- [API e inferência](#api-e-inferência)
- [Docker](#docker)
- [Hiperparâmetros](#hiperparâmetros)
- [Testes e lint](#testes-e-lint)
- [Dataset](#dataset)
- [Vídeo STAR](#vídeo-star)
- [Créditos](#créditos)

## Visão geral

| Item | Valor |
|---|---|
| Problema | Triagem automática de laudos médicos por nível de urgência |
| Dataset | [Medical Abstracts TC Corpus](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus) (14.438 registros) |
| Modelo | TF-IDF (20k features, 1-2 ngrams, English stopwords) + LogisticRegression (C=0.5, balanced) + ajuste de limiares por classe |
| Métricas | Accuracy 77.7%, F1 macro 72.0%, Recall macro 80.6%, Recall urgente 80.7% |
| Otimização | Exportação ONNX (artefato 99.7% menor, 37.44 MB → 115.8 KB), paridade 100% |
| Orquestração | DVC (4 stages: ingestão → treino → avaliação → ONNX) + Airflow (`@weekly`) |
| Tracking | DVC (parâmetros, métricas, artefatos versionados) |
| CI/CD | GitHub Actions (lint + test + build/push GHCR) |
| Monitoramento | Prometheus (4 métricas) + Grafana (4 painéis auto-provisionados) |
| Container | Docker multi-stage (`python:3.14-slim`), compose com perfis (api, train, airflow, monitoring) |
| Build | uv + `pyproject.toml` com deps prod/dev separadas |
| Lint | ruff + pre-commit |
| Testes | pytest (129 testes) |

## Arquitetura

```mermaid
flowchart TD
    A["Medical Abstracts TC Corpus\n(Kaggle, 14.438 registros)"] --> B["DVC pipeline\ningestão → treino\n → avaliação → onnx"]
    B --> C["DVC Tracking\n(parâmetros, métricas,\n artefatos versionados)"]
    B --> D["models/model.joblib\n+\nmodel.onnx"]
    D --> E["FastAPI\n/health /metrics\n /predict /predict /batch"]
    E --> F["Prometheus\napp_requests_total\napp_request_latency_seconds\npredictions_total\nprediction_latency_seconds"]
    F --> G["Grafana\nTaxa de Requisições\nLatência P95\nTaxa de Erro\nTotal de Requisições"]
    H["Airflow @weekly\ntrain_pipeline DAG"] --> B
```

Design patterns aplicados:

- **Factory** — `src/models/factory.py` instancia `JoblibLoader` ou `OnnxLoader` por nome de backend.
- **Strategy** — normalizador de texto intercambiável via `TextNormalizer`; classificador configurável via `params.yaml` (RandomForest ou LogisticRegression).
- **Atomic file writes** — todos os artefatos usam write-to-temp-then-rename (`src/core/dataset.py`).

## Entregáveis por etapa

| Etapa | Requisito do edital | Status | Localização |
|---|---|---|---|
| 1 | Decisão arquitetural (AWS ECS/Fargate) documentada no README | ✔ Concluído | este README, seção [Decisão arquitetural](#decisão-arquitetural) |
| 1 | API FastAPI funcional em Docker | ✔ Concluído | `src/api/app.py`, `docker/Dockerfile`, `docker/docker-compose.yml` |
| 1 | Baseline de latência local | ✔ Concluído | `scripts/benchmark.py`, seção [Otimização de latência](#otimização-de-latência) |
| 2 | GitHub Actions workflow (lint + test + build) | ✔ Concluído | `.github/workflows/ci.yml` |
| 2 | DAG Airflow funcional (ingestão → treino → avaliação → exportação ONNX) | ✔ Concluído | `dags/train_pipeline.py`, `src/orchestration/training_tasks.py` |
| 2 | Pipeline DVC reprodutível | ✔ Concluído | `dvc.yaml` (4 stages), `dvc.lock` |
| 3 | Instrumentação Prometheus (`prometheus_client`) | ✔ Concluído | `src/core/metrics.py`, `src/api/app.py` (`/metrics`) |
| 3 | Docker Compose com API + Prometheus + Grafana | ✔ Concluído | `docker/docker-compose.yml` (perfil `monitoring`) |
| 3 | Dashboard Grafana com ≥3 painéis | ✔ Concluído | `configs/grafana/dashboards/triage-api.json` (Taxa de Requisições, Latência P95, Taxa de Erro, Total de Requisições) |
| 4 | Modelo treinado (TF-IDF + LogisticRegression + limiares) | ✔ Concluído | `models/model.joblib`, `models/thresholds.json`, `configs/params.yaml` |
| 4 | Otimização ONNX + comparativo de latência | ✔ Concluído | `models/model.onnx`, `reports/benchmark_comparison.md` |
| 4 | README completo cobrindo todos os entregáveis | ✔ Concluído | este README |
| 4 | Vídeo STAR de 5 minutos | ✔ Concluído | seção [Vídeo STAR](#vídeo-star) |

## Critérios de avaliação

| Critério | Peso | Status | Evidência |
|---|---|---|---|
| Modelagem/Otimização | 20% | ✔ Concluído | TF-IDF + LogisticRegression com ajuste de limiares, ONNX exportado (paridade com limiares), benchmark comparativo, modelo real (14.438 registros, F1 macro 72.0%, recall macro 80.6%) |
| CI/CD (GitHub Actions) | 15% | ✔ Concluído | `.github/workflows/ci.yml`: lint → test → build/push GHCR em push/PR para `main` |
| Orquestração (Airflow) | 15% | ✔ Concluído | DAG `train_pipeline` (`@weekly`, 5 tasks TaskFlow), stack Airflow via Docker Compose |
| Monitoramento | 20% | ✔ Concluído | 4 métricas Prometheus, `/metrics`, Grafana 4 painéis auto-provisionados, `generate_traffic.py` |
| README | 15% | ✔ Concluído | este documento — arquitetura, instruções passo a passo, latência, FinOps, segurança, CI/CD |
| Vídeo STAR | 15% | ✔ Concluído | seção [Vídeo STAR](#vídeo-star) |

## Bibliotecas requeridas

| Biblioteca | Versão | Uso | Localização |
|---|---|---|---|
| **scikit-learn** | ≥1.6.0 | Modelo base de classificação de texto (TF-IDF + LogisticRegression + ajuste de limiares) | `src/train/run.py`, `src/train/thresholds.py`, `configs/params.yaml` |
| **FastAPI** | ≥0.115.0 | API REST para inferência (`/predict`, `/predict/batch`, `/health`, `/metrics`) | `src/api/app.py` |
| **prometheus-client** | ≥0.21.0 | Instrumentação de métricas (latência, contagem de requisições e predições) | `src/core/metrics.py`, `/metrics` endpoint |
| **apache-airflow** | ==3.3.1 | Orquestração de retreino (DAG `train_pipeline`, `@weekly`) | `dags/train_pipeline.py`, `src/orchestration/training_tasks.py` |

> Dependências de otimização: `onnxruntime` (inferência ONNX), `skl2onnx` (conversão sklearn → ONNX). Declaradas em `pyproject.toml`.

## Estrutura do projeto

```
.
├── src/
│   ├── api/                # FastAPI (app, schemas, rate limiting)
│   ├── core/               # config, dataset, metrics, params, stopwords
│   ├── evaluate/           # avaliação (classification report, confusion matrix)
│   ├── models/             # factory (JoblibLoader, OnnxLoader), urgency enum
│   ├── orchestration/      # Airflow training tasks
│   ├── preprocess/         # normalização de texto, pipeline de ingestão
│   ├── train/              # treino (LogisticRegression/RandomForest), limiares, exportação ONNX
│   └── validate/           # validação de dados brutos
├── tests/                  # 129 testes pytest
├── dags/                   # Airflow DAG (train_pipeline)
├── docker/                 # Dockerfile, docker-compose.yml, airflow.Dockerfile
├── configs/                # params.yaml, prometheus.yml, Grafana provisioning
├── scripts/                # benchmark, download_data, generate_synthetic_data, generate_traffic, validate_env
├── data/raw/               # laudos.csv (DVC tracked)
├── data/processed/         # laudos_processed.csv, test_split.csv (DVC tracked)
├── models/                 # model.joblib, model.onnx + hashes SHA256, thresholds.json
├── reports/                # classification_report.txt, metrics.json, benchmark_comparison.md
├── dvc.yaml                # 4 stages: ingestão → treino → avaliação → onnx
├── dvc.lock
├── Makefile                # atalhos para setup/lint/test/pipeline/benchmark/airflow/monitoring
└── pyproject.toml          # uv + deps prod/dev
```

## Pré-requisitos

- Python ≥ 3.14
- [uv](https://docs.astral.sh/uv/) (gerenciador de dependências)
- [DVC](https://dvc.org/) (instalado via `uv sync`)
- Docker + Docker Compose (para API, Airflow e monitoramento)
- Conta Kaggle + token (para `make data-kaggle`)

## Setup rápido

```bash
# 1. Clonar e entrar na pasta
git clone https://github.com/G13-MLE/tech-challenge-fase-3.git
cd tech-challenge-fase-3

# 2. Copiar variáveis de ambiente e preencher credenciais
cp .env.example .env
#   Edite .env e preencha:
#     - KAGGLE_USERNAME, KAGGLE_KEY          (para baixar o dataset)
#     - DVC_ONEDRIVE_REMOTE_URL              (opcional, para compartilhar artefatos)

# 3. Sincronizar dependências e configurar pre-commit + DVC remote
make setup

# 4. Baixar o dataset real (Medical Abstracts TC Corpus, 14.438 registros)
make data-kaggle

# 5. Rodar o pipeline completo
make pipeline

# 6. Subir a API em Docker
make docker-run
#  API disponível em http://localhost:8000
```

> **Dataset sintético**: para desenvolvimento rápido sem Kaggle, use `make data-synthetic` (180 laudos, 3 classes, seed=42).

## Como executar o pipeline

### Caminho A — pipeline reprodutível via DVC

```bash
# Baixar dataset real e registrar no DVC
make data-kaggle

# Executar os 4 stages com saída ao vivo:
#   ingestão  -> data/processed/laudos_processed.csv
#   treino    -> models/model.joblib + data/processed/test_split.csv + reports/
#   avaliação -> reports/metrics.json + reports/classification_report.txt
#   onnx      -> models/model.onnx + models/model.onnx.sha256
make pipeline

# Para refazer tudo do zero (ignora cache DVC):
uv run dvc repro --force -v

# Persistir artefatos no remote DVC (OneDrive, opcional):
make dvc-remote   # configura o remote (se DVC_ONEDRIVE_REMOTE_URL estiver preenchido)
uv run dvc push
```

### Caminho B — pipeline direto (mais rápido para iteração)

```bash
make pipeline-live     # tudo de uma vez
make train-live        # apenas o treino
make evaluate-live     # apenas a avaliação
```

## Decisão arquitetural

### Estratégia de deploy: Real-time (AWS ECS/Fargate)

**Cenário**: triagem clínica exige resposta imediata — o laudo não pode esperar um job agendado.

| Estratégia | Latência | Custo | Complexidade | Adequação |
|---|---|---|---|---|
| Batch | Alta, tolerada | Menor | Baixa | Não — triagem não espera job |
| Real-time | Baixa, determinística | Maior (infra sempre ativa) | Alta | **Escolhida** |
| Serverless | Variável (cold start) | Eficiente em cargas intermitentes | Média | Alternativa citada; cold start impacta latência |

**Escolha**: AWS ECS/Fargate com container Docker.

- Modelo leve (TF-IDF + LogisticRegression, 37.44 MB joblib / 115.8 KB ONNX) carregado em memória no startup
- Escalabilidade horizontal via ECS, sem gerenciamento de servidores (Fargate)
- Latência previsível e baixa (P50 ≈ 12.03 ms), requisito crítico para triagem

### Tecnologia

| Componente | Decisão |
|---|---|
| API | FastAPI (`/health`, `/metrics`, `/api/v1/predict`, `/api/v1/predict/batch`) |
| Modelo | Scikit-Learn (TF-IDF + LogisticRegression + limiares), artefato em `joblib` + `onnx` |
| Container | Docker multi-stage, `python:3.14-slim`, não-root (`appuser`) |
| Benchmark | Percentis P50/P95/P99 (nunca média), comparativo joblib vs ONNX |
| Orquestração | DVC (pipeline reprodutível) + Airflow (re-treino semanal) |
| CI/CD | GitHub Actions (lint → test → build/push GHCR) |
| Monitoramento | Prometheus + Grafana (4 painéis auto-provisionados) |

### Mapeamento de urgência

O modelo classifica o texto em 5 classes clínicas do Medical Abstracts TC Corpus, mapeadas para 3 níveis de urgência:

| Classe clínica (condition_label) | Urgência | Label |
|---|---|---|
| Neoplasms (1) | Urgente | 2 |
| Cardiovascular diseases (4), Nervous system diseases (3), General pathological conditions (5) | Atenção | 1 |
| Digestive system diseases (2) | Normal | 0 |

Distribuição do dataset: 1.494 normal (10,3%), 9.781 atenção (67,7%), 3.163 urgente (21,9%), total 14.438 registros (80/20 split).

> **Dataset sintético** (`make data-synthetic`): fallback para desenvolvimento sem
> Kaggle, gera 180 laudos em português com 3 classes de urgência diretas
> (0=normal: asma/hérnia, 1=atenção: diabetes/hipertensão, 2=urgente: pneumonia).
> O esquema de saída (colunas `text`, `label` ∈ {0,1,2}) é idêntico ao dataset real,
> permitindo rodar o pipeline completo sem credenciais Kaggle.

## Otimização de latência

### Resultados com modelo real (Medical Abstracts TC, 14.438 registros)

Benchmark Docker, 100 requisições sequenciais, warmup 10, rate limiting desabilitado:

#### Baseline (joblib)

| Percentil | Latência |
|---|---|
| P50 | 12.03 ms |
| P95 | 13.82 ms |
| P99 | 45.89 ms |

#### Comparativo joblib vs ONNX

| Métrica | joblib | ONNX | Δ |
|---|---|---|---|
| P50 | 12.03 ms | 8.80 ms | -3.23 ms (-26.8%) |
| P95 | 13.82 ms | 13.02 ms | -0.80 ms (-5.8%) |
| P99 | 45.89 ms | 21.35 ms | -24.54 ms (-53.5%) |
| min | 10.33 ms | 6.93 ms | -3.40 ms (-32.9%) |
| max | 46.12 ms | 21.37 ms | -24.76 ms (-53.7%) |
| Artefato | 37.44 MB | 115.8 KB | **ONNX é 99.7% menor** (0.3% do tamanho joblib) |

> Relatório completo: `reports/benchmark_comparison.md`
>
> **Nota**: P50 e P99 do ONNX são inferiores ao joblib (melhoria de 26.8% e 53.5%
> respectivamente); P95 do ONNX é marginalmente inferior (-5.8%) dentro da
> variabilidade esperada. A paridade de predição entre joblib e ONNX é de 100%
> (threshold de 95%).

### Métricas do modelo (teste, 2.888 amostras)

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.42 | 0.45 | 0.43 | 299 |
| atenção | 0.86 | 0.81 | 0.83 | 1.956 |
| urgente | 0.62 | 0.71 | 0.66 | 633 |
| **Macro avg** | **0.63** | **0.66** | **0.64** | 2.888 |
| **Weighted avg** | **0.76** | **0.75** | **0.75** | 2.888 |

- Accuracy: 75.0%
- CV F1 macro (5-fold): 0.650 ± 0.008
- Modelo com `class_weight=balanced` para compensar desbalanceamento

## FinOps — Estimativa de custos AWS

### Cenário: ECS Fargate (produção, tráfego moderado)

| Recurso | Especificação | Custo mensal (USD) |
|---|---|---|
| ECS Fargate (1 task) | 0.5 vCPU, 1 GB RAM, 24/7 | ≈ 27 |
| Application Load Balancer | 1 ALB + processamento | ≈ 25 |
| NAT Gateway | 1 gateway + 10 GB data | ≈ 35 |
| ECR | 2 imagens × ~200 MB | ≈ 1 |
| CloudWatch | Logs + métricas básicas | ≈ 5 |
| Data transfer | 50 GB/mês out | ≈ 4 |
| Secrets Manager | 2 secrets (API key, Kaggle) | ≈ 1 |
| **Total estimado** | | **≈ 98 USD/mês** |

### Cenários alternativos

| Cenário | ECS tasks | Custo estimado |
|---|---|---|
| Desenvolvimento (1 task, sem ALB) | 1 | ≈ 30 USD/mês |
| Produção baixo tráfego | 1 + ALB | ≈ 55 USD/mês |
| Produção médio tráfego (2 tasks, multi-AZ) | 2 + ALB | ≈ 98 USD/mês |
| Produção alto tráfego (4 tasks, auto-scaling) | 4 + ALB | ≈ 150 USD/mês |

### Estratégias de otimização de custo

- **Auto-scaling ECS**: escalar para 0 fora do horário de pico (triagem não é 24/7 em muitos hospitais)
- **Spot instances/Fargate Spot**: até 70% de desconto para tasks tolerantes a interrupção
- **Graviton2 (ARM64)**: até 20% melhor custo/desempenho que x86
- **Reserva de capacity**: Savings Plans para cargas previsíveis (1 ano: ≈ 20% desconto)
- **Modelo leve**: 115.8 KB ONNX permite instâncias menores (0.25 vCPU) sem impacto perceptível

## Segurança

### Rate limiting

- Implementado em `src/core/ratelimit.py` (sliding window por IP)
- Configurável via `.env`: `RATE_LIMIT_MAX_PER_IP=60` requisições por `RATE_LIMIT_WINDOW_SECONDS=60`
- Desabilitado em benchmark (`RATE_LIMIT_MAX_PER_IP=0`)
- Proteção contra abuso e ataques de força bruta

### Menor privilégio (IAM e container)

- Container Docker executa como `appuser` (não-root)
- IAM ECS task role com permissões mínimas: apenas leitura do ECR e acesso ao Secrets Manager para API key
- Secret `kaggle_key` usa `SecretStr` no Pydantic Settings (não exposta em logs)
- API key opcional (`X-API-Key` header) para autenticação

### Proteção contra Model Extraction

- Rate limiting (60 req/min por IP) dificulta extração sistemática do modelo
- Endpoint `/predict` retorna apenas o nível de urgência e confiança — nunca probabilidades brutas de todas as classes
- Não expõe hiperparâmetros, vocabulário TF-IDF ou metadados internos
- Health check (`/health`) não vaza informações do modelo

## CI/CD

Workflow GitHub Actions (`.github/workflows/ci.yml`) com 3 jobs:

```
push/PR → main
    │
    ├── lint   ── ruff check + ruff format --check
    ├── test   ── pytest (129 testes)
    └── build  ── Docker build + push para ghcr.io (após lint+test)
```

| Job | Trigger | Passos |
|---|---|---|
| `lint` | push/PR para `main` | `uv sync --frozen` → `ruff check` → `ruff format --check` |
| `test` | push/PR para `main` | `uv sync --frozen` → `pytest` |
| `build` | push/PR para `main` (após lint+test) | Docker build → push `ghcr.io/<repo>/triage-api:<sha>` (skip push em PRs) |

- Concorrência: `cancel-in-progress` por branch
- Permissões: `contents: read` (lint/test), `packages: write` (build)
- Cache: `setup-uv@v6` com cache do `uv.lock`

## Histórico de commits

O repositório segue **Conventional Commits** com branches organizadas por etapa do challenge:

| Tipo | Uso | Exemplos |
|---|---|---|
| `feat:` | Nova funcionalidade | `feat: integra Prometheus à stack de monitoramento` |
| `ci:` | Configuração de CI/CD | `ci: add GitHub Actions workflow (lint + test + build)` |
| `setup:` | Configuração inicial | `setup: configura fundação do projeto` |
| `refactor:` | Refatoração sem mudança de comportamento | `refactor: torna caminhos de avaliação injetáveis` |
| `docs:` | Documentação | `docs: atualiza métricas de benchmark` |

Estrutura de branches por etapa: `etapa-1/...`, `etapa-2/airflow`, `etapa-3/prometheus`, `etapa-4/...`. Cada etapa é desenvolvida em branch separada e mesclada via PR, garantindo rastreabilidade.

```bash
git log --oneline
# d952c0c Initial commit
# 2e859e8 setup: configura fundação do projeto (#1)
# 78bd566 feat: arquitetura api inicial (#9)
# d682732 feat: implementa pipeline DVC (#13)
# 2a7d2f3 feat: adiciona DAG Airflow (#14)
# 2cf6f27 ci: add GitHub Actions workflow (#15)
# 6aef044 feat: integra Prometheus (#16)
# 5d50d97 feat: adiciona Grafana (#17)
# e6dbf21 feat: exporta modelo ONNX (#18)
# 963986e feat: consolida etapa 4 (#19)
```

## Airflow

Stack containerizada: Airflow 3.x com `LocalExecutor` (PostgreSQL para metadados, sem Redis/Celery).

Containers: `airflow-postgres`, `airflow-init`, `airflow-scheduler`, `airflow-dag-processor`, `airflow-apiserver` (UI + API).

### DAG `train_pipeline`

```python
# dags/train_pipeline.py — schedule="@weekly", TaskFlow API
(
    ingest_data_task()
    >> train_model_task()
    >> save_model_task()
    >> evaluate_model_task()
    >> export_onnx_task()
)
```

| Propriedade | Valor |
|---|---|
| Schedule | `@weekly` |
| Start date | 2026-01-01 |
| Catchup | False |
| Max active runs | 1 |
| Tasks | ingest_data → train_model → save_model → evaluate_model → export_onnx |
| Retries | 3 (ingestão/treino), 2 (validação/avaliação/exportação) |
| Timeout | 1h (ingestão/treino), 30min (validação/avaliação/exportação) |

### Subir a stack

```bash
make airflow-up
# Airflow UI: http://localhost:8080
make airflow-down   # parar
```

## Monitoramento

Stack Prometheus + Grafana containerizada. A API expõe `/metrics` com 4 métricas Prometheus; o Grafana provisiona automaticamente datasource e dashboard.

### Métricas instrumentadas

| Métrica | Tipo | Descrição |
|---|---|---|
| `app_requests_total` | Counter | Total de requisições por método/endpoint |
| `app_request_latency_seconds` | Histogram | Latência de requisições |
| `predictions_total` | Counter | Predições por nível de urgência |
| `prediction_latency_seconds` | Histogram | Latência de predição por endpoint |

### Dashboard Grafana (4 painéis)

1. **Taxa de Requisições** — `rate(app_requests_total[5m])`
2. **Latência P95** — `histogram_quantile(0.95, rate(app_request_latency_seconds_bucket[5m]))`
3. **Taxa de Erro** — `rate(app_requests_total{http_status=~"[45].."}[5m])`
4. **Total de Requisições** — `sum(app_requests_total)`

### Subir a stack

```bash
make monitoring-up
make generate-traffic   # popular métricas
# API:        http://localhost:8000
# Métricas:   http://localhost:8000/metrics
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin / $GRAFANA_ADMIN_PASSWORD)
make monitoring-down     # parar
```

## API e inferência

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Health check (ok / degraded) |
| GET | `/metrics` | Métricas Prometheus |
| POST | `/api/v1/predict` | Classificação de urgência |
| POST | `/api/v1/predict/batch` | Classificação em lote (até 100 textos) |

### Exemplos

```bash
# Predição individual
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "patient with severe bacterial pneumonia requiring immediate ventilation"}'

# Predição em lote
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["neoplasm metastasis", "chronic hypertension", "mild digestive discomfort"]}'

# Health check
curl http://localhost:8000/health

# Métricas Prometheus
curl http://localhost:8000/metrics
```

Autenticação opcional por API key (`X-API-Key`): habilite via `API_KEY_ENABLED=true` e defina `API_KEY` no `.env`.

### Benchmark de latência

```bash
make benchmark               # benchmark com backend atual
make benchmark-comparison     # comparativo joblib vs ONNX
```

## Docker

### Imagem da API (multi-stage)

```bash
make docker-build            # construir imagem
make docker-run              # rodar API em container
make docker-train            # pipeline DVC no container (perfil train)
```

### Stack completa (perfis Docker Compose)

```bash
make airflow-up              # Airflow (perfil airflow)
make monitoring-up           # API + Prometheus + Grafana (perfil monitoring)
make api-up                  # API standalone (perfil default)
```

## Hiperparâmetros

Todos os hiperparâmetros ficam em [`configs/params.yaml`](configs/params.yaml) e são lidos pelo DVC:

| Bloco | Parâmetro | Default | Descrição |
|---|---|---|---|
| `train` | `seed` | 42 | Seed reprodutível |
| `train` | `test_size` | 0.2 | Proporção do split de teste |
| `train` | `classifier` | `logistic_regression` | Classificador: `random_forest` ou `logistic_regression` |
| `train` | `random_forest_n_estimators` | 100 | Número de árvores (usado se classifier=random_forest) |
| `train` | `random_forest_class_weight` | `balanced` | Peso das classes do RandomForest (compensa desbalanceamento) |
| `train` | `logistic_regression_C` | 0.5 | Inverso da regularização da regressão logística |
| `train` | `logistic_regression_class_weight` | `balanced` | Peso das classes da regressão logística |
| `train` | `logistic_regression_max_iter` | 2000 | Número máximo de iterações da regressão logística |
| `train` | `tfidf_max_features` | 20000 | Máximo de features TF-IDF |
| `train` | `tfidf_ngram_range` | [1, 2] | N-grams (1=unigrams, 2=bigrams) |
| `train` | `tfidf_sublinear_tf` | true | Log1p na frequência de termos |
| `train` | `tfidf_min_df` | 2 | Frequência mínima de documento |
| `train` | `tfidf_max_df` | 0.95 | Frequência máxima de documento |
| `train` | `tfidf_stopwords` | true | Stopwords (true=inglês, false=sem) |
| `train` | `cv_folds` | 5 | Folds de cross-validation (0=desativado) |
| `train` | `threshold_tuning` | true | Ajuste de limiares por classe via CV (maximiza recall macro) |
| `train` | `threshold_min_f1` | 0.55 | F1 macro mínimo ao buscar limiares (trade-off recall vs precisão) |

## Testes e lint

```bash
make test      # uv run pytest (129 testes)
make lint      # uv run ruff check . + ruff format --check .
make format    # uv run ruff format .
```

## Dataset

**Medical Abstracts TC Corpus** ([Kaggle](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus)):

| Arquivo | Registros | Conteúdo |
|---|---|---|
| `medical_tc_train.csv` | 11.550 | `condition_label` (1-5) + `medical_abstract` (texto) |
| `medical_tc_test.csv` | 2.888 | `condition_label` (1-5) + `medical_abstract` (texto) |
| `medical_tc_labels.csv` | 5 | Mapeamento condition_label → condition_name |

Mapeamento condition_label → urgência:

| condition_label | condition_name | Urgência |
|---|---|---|
| 1 | Neoplasms | Urgente (2) |
| 3 | Nervous system diseases | Atenção (1) |
| 4 | Cardiovascular diseases | Atenção (1) |
| 5 | General pathological conditions | Atenção (1) |
| 2 | Digestive system diseases | Normal (0) |

Os CSVs são baixados via `make data-kaggle`, processados e salvos como `data/raw/laudos.csv` (colunas `text`, `label`). Não são commitados no Git (ver `.gitignore`). Use `make data-kaggle` para baixar e `uv run dvc push` para sincronizar com o remote OneDrive.

## Vídeo STAR

> **Link:** vídeo finalizado e entregue.
>
> Roteiro esperado (≤ 5 minutos):
> - **Situation**: Hospital de referência precisa de triagem automática de laudos médicos
> - **Task**: Requisitos da fase (latência < 50ms, CI/CD, Airflow, monitoramento)
> - **Action**: Arquitetura ECS/Fargate, pipeline DVC 4 stage, ONNX para otimização, Prometheus+Grafana
> - **Result**: Demo do pipeline funcionando, latência P95=13.82ms (joblib), P99 ONNX 53.5% mais rápido, dashboard Grafana, CI verde

## Créditos

**G13-MLE** — Grupo 13 (PÓS TECH FIAP) - Tech Challenge Fase 03:

- Eduardo Nunes Pereira

## Licença

Uso acadêmico restrito aos participantes do Tech Challenge FIAP.
