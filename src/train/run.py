"""Pipeline de treinamento do modelo de triagem.

Este módulo será implementado na Etapa 4 com a lógica de treinamento
do modelo real (TF-IDF + RandomForest) usando o dataset completo,
registrando métricas no MLflow e salvando o artefato em joblib/onnx.
"""

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Executa o pipeline de treinamento.

    Stub reservado para a Etapa 4. Quando implementado, deverá:
    - Carregar os dados processados de data/processed/
    - Treinar o Pipeline TF-IDF + RandomForest
    - Avaliar com métricas (acurácia, F1, precision, recall)
    - Registrar no MLflow
    - Salvar o artefato em models/model.joblib
    """
    logger.info("Treinamento: stub não implementado (Etapa 4).")


if __name__ == "__main__":
    main()
