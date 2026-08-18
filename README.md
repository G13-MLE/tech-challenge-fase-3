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
| Modelo | Scikit-Learn (TF-IDF + RandomForest), artefato em `joblib` |
| Container | Docker multi-stage, `python:3.14-slim` |
| Benchmark | Percentis P50/P95/P99 (nunca média) |

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

> Nota: na Etapa 1 a API usa um modelo dummy em `joblib` com esse mapeamento. O modelo real é treinado na Etapa 4.

## Início Rápido

### Requisitos

- Python 3.14+
- uv (gerenciador de pacotes)
- Docker (opcional, para container)

### Instalação

```bash
make setup
```

### Gerar modelo dummy

```bash
make create-model
```

### Executar localmente

```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Executar em Docker

```bash
make docker-run
```

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Classificação de urgência |

Exemplo:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "paciente com pneumonia bacteriana grave"}'
```

### Benchmark de latência

```bash
make benchmark
```

## Baseline de Latência

Resultados obtidos com `make benchmark` em Docker (modelo dummy, 100 requisições):

| Percentil | Latência |
|---|---|
| P50 | 9.61 ms |
| P95 | 15.39 ms |
| P99 | 31.35 ms |

## Comandos Disponíveis

```bash
make help
```

| Comando | Descrição |
|---|---|
| `make setup` | Configurar ambiente |
| `make test` | Rodar testes |
| `make lint` | Verificar código |
| `make create-model` | Gerar modelo dummy |
| `make benchmark` | Benchmark de latência |
| `make docker-build` | Construir imagem Docker |
| `make docker-run` | Rodar API em container |
