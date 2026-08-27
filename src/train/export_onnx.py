"""Exportação do modelo sklearn para formato ONNX.

Converte o Pipeline TF-IDF + classificador serializado em joblib para
ONNX via skl2onnx, executa teste de paridade de predição (joblib vs ONNX)
e salva o artefato com hash SHA256 para verificação de integridade.
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from src.core.dataset import (
    ONNX_MODEL_HASH_PATH,
    ONNX_MODEL_PATH,
    TEST_DATA_PATH,
    atomic_write_bytes,
    atomic_write_text,
    load_csv_records,
)

logger = logging.getLogger(__name__)

__all__ = ["export_onnx", "verify_onnx_parity"]


def export_onnx(
    model_path: Path = Path("models/model.joblib"),
    onnx_path: Path = ONNX_MODEL_PATH,
    onnx_hash_path: Path = ONNX_MODEL_HASH_PATH,
    target_opset: int = 15,
) -> Path:
    """Exporta o modelo sklearn para ONNX e salva com hash SHA256.

    Args:
        model_path: Caminho do modelo joblib.
        onnx_path: Caminho de saída do modelo ONNX.
        onnx_hash_path: Caminho do arquivo de hash SHA256.
        target_opset: Versão do opset ONNX alvo (15 é compatível com
            onnxruntime>=1.16, que é a versão usada no container Docker).

    Returns:
        Caminho do modelo ONNX salvo.
    """
    import joblib as joblib_mod

    pipeline = joblib_mod.load(model_path)

    n_features = 1
    initial_types = [("input", StringTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset=target_opset,
        options={id(pipeline): {"zipmap": False}},
    )

    onnx_bytes = onnx_model.SerializeToString()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(onnx_path, onnx_bytes)

    sha256 = hashlib.sha256(onnx_bytes).hexdigest()
    atomic_write_text(onnx_hash_path, sha256)

    logger.info("Modelo ONNX exportado para %s (%d bytes)", onnx_path, len(onnx_bytes))
    logger.info("Hash SHA256: %s...", sha256[:16])

    return onnx_path


PARITY_THRESHOLD = 0.95


def verify_onnx_parity(
    model_path: Path = Path("models/model.joblib"),
    onnx_path: Path = ONNX_MODEL_PATH,
    test_data_path: Path = TEST_DATA_PATH,
    max_samples: int = 100,
    probability_tolerance: float = 1e-4,
) -> bool:
    """Verifica paridade de predição entre o modelo joblib e ONNX.

    Compara as predições (argmax e probabilidade) dos dois backends
    sobre uma amostra dos dados de teste. Diferenças pequenas são esperadas
    devido à normalização de texto (StringNormalizer) no ONNX Runtime.

    Args:
        model_path: Caminho do modelo joblib.
        onnx_path: Caminho do modelo ONNX.
        test_data_path: Caminho do CSV de teste.
        max_samples: Número máximo de amostras para teste.
        probability_tolerance: Tolerância absoluta para diferença de probabilidade.

    Returns:
        True se a taxa de acerto >= PARITY_THRESHOLD, False caso contrário.
    """
    import joblib as joblib_mod
    import onnxruntime as ort

    from src.models.urgency import URGENCY_MAP

    pipeline = joblib_mod.load(model_path)
    session = ort.InferenceSession(onnx_path.read_bytes(), providers=["CPUExecutionProvider"])

    texts, _ = load_csv_records(test_data_path)
    texts = texts[:max_samples]

    joblib_probas = pipeline.predict_proba(texts)
    joblib_preds = pipeline.predict(texts)

    input_name = session.get_inputs()[0].name
    onnx_inputs = {input_name: np.array(texts).reshape(-1, 1)}
    onnx_outputs = session.run(None, onnx_inputs)
    onnx_labels = onnx_outputs[0]
    onnx_probas = onnx_outputs[1]

    mismatches = 0
    for i in range(len(texts)):
        joblib_label = int(joblib_preds[i])
        onnx_label = int(onnx_labels[i])
        if joblib_label != onnx_label:
            logger.warning(
                "Paridade: amostra %d — joblib=%s(%s), onnx=%s(%s)",
                i,
                joblib_label,
                URGENCY_MAP.get(joblib_label, "?"),
                onnx_label,
                URGENCY_MAP.get(onnx_label, "?"),
            )
            mismatches += 1

        joblib_conf = float(joblib_probas[i].max())
        onnx_conf = float(onnx_probas[i].max())
        if abs(joblib_conf - onnx_conf) > probability_tolerance:
            logger.warning(
                "Paridade: amostra %d — confiança joblib=%.6f, onnx=%.6f (diff=%.6f)",
                i,
                joblib_conf,
                onnx_conf,
                abs(joblib_conf - onnx_conf),
            )

    match_rate = 1 - mismatches / len(texts)
    if match_rate < PARITY_THRESHOLD:
        logger.error(
            "Paridade falhou: %d/%d predições divergem (taxa de acerto=%.1f%%, mínimo=%.0f%%)",
            mismatches,
            len(texts),
            match_rate * 100,
            PARITY_THRESHOLD * 100,
        )
        return False

    logger.info(
        "Paridade OK: %d/%d predições consistentes (taxa de acerto=%.1f%%)",
        len(texts) - mismatches,
        len(texts),
        match_rate * 100,
    )
    return True


def main() -> None:
    """Exporta o modelo para ONNX e verifica a paridade de predição."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Exportando modelo joblib para ONNX...")
    onnx_path = export_onnx()

    logger.info("Verificando paridade de predição joblib vs ONNX...")
    parity_ok = verify_onnx_parity()
    if not parity_ok:
        raise RuntimeError("Teste de paridade joblib vs ONNX falhou!")

    logger.info("Exportação ONNX concluída com sucesso: %s", onnx_path)


if __name__ == "__main__":
    main()
