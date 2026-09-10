"""Factory Pattern para carregamento de modelos de ML.

Disponibiliza uma interface uniforme para carregar artefatos de modelo
a partir de diferentes backends (joblib, onnx), encapsulando a lógica
de desserialização, verificação de integridade e inferência.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import joblib
import numpy as np

from src.core.dataset import EXPECTED_CLASSES
from src.models.urgency import URGENCY_MAP, UrgencyLevel
from src.train.thresholds import DEFAULT_THRESHOLDS, apply_thresholds, load_thresholds

__all__ = ["ModelLoader", "JoblibLoader", "OnnxLoader", "create_model", "register_loader"]

logger = logging.getLogger(__name__)

ModelBackend = Literal["joblib", "onnx"]

_LOADERS: dict[str, type[ModelLoader]] = {}


def register_loader(name: str):
    """Decorator para registrar um ModelLoader no registry.

    Args:
        name: Nome do backend para registro.

    Returns:
        Decorator que registra a classe e a retorna inalterada.
    """

    def decorator(cls: type[ModelLoader]) -> type[ModelLoader]:
        _LOADERS[name] = cls
        return cls

    return decorator


class ModelLoader(ABC):
    """Interface abstrata para carregadores de modelo.

    Cada implementação define como um artefato de modelo é carregado,
    verificado e como a inferência é realizada sobre textos de entrada.
    """

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Carrega o modelo do caminho especificado.

        Args:
            path: Caminho para o artefato de modelo.
        """

    @abstractmethod
    def predict(self, text: str) -> tuple[UrgencyLevel, float]:
        """Realiza a inferência sobre o texto de entrada.

        Args:
            text: Texto do laudo médico.

        Returns:
            Tupla (urgência, confiança) com o nível de urgência
            e a probabilidade associada à predição.
        """

    def predict_batch(self, texts: list[str]) -> list[tuple[UrgencyLevel, float]]:
        """Realiza inferência em lote sobre múltiplos textos.

        Implementação padrão itera sobre os textos. Subclasses podem
        sobrescrever para usar vetorização nativa do backend.

        Args:
            texts: Lista de textos de laudos médicos.

        Returns:
            Lista de tuplas (urgência, confiança).
        """
        return [self.predict(text) for text in texts]


@register_loader("joblib")
class JoblibLoader(ModelLoader):
    """Carregador de modelos serializados com joblib.

    Espera um Pipeline sklearn com TfidfVectorizer e um classificador
    que implemente predict_proba. As classes do modelo devem mapear
    para os níveis de urgência na ordem: normal, atenção, urgente.

    Inclui verificação de integridade SHA256 do artefato e validação
    das classes do modelo contra o mapeamento esperado.

    Opcionalmente carrega limiares de decisão de um arquivo sidecar
    (`<model_path>.thresholds.json` ou `models/thresholds.json`) para
    ajustar as predições e maximizar o recall macro.
    """

    def __init__(self) -> None:
        self._model = None
        self._thresholds: dict[str, float] = dict(DEFAULT_THRESHOLDS)

    def load(self, path: Path | str) -> None:
        """Carrega o Pipeline sklearn do arquivo joblib com verificação de integridade.

        Args:
            path: Caminho para o arquivo .joblib.

        Raises:
            FileNotFoundError: Se o arquivo de modelo não existir.
            ValueError: Se o hash SHA256 não corresponder ou as classes
                do modelo não forem as esperadas.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {path}")
        _verify_model_hash(path)
        self._model = joblib.load(path)
        _validate_model_classes(self._model)
        self._thresholds = _load_thresholds_for_model(path)

    def predict(self, text: str) -> tuple[UrgencyLevel, float]:
        """Prediz o nível de urgência e a confiança associada.

        Args:
            text: Texto do laudo médico (deve estar normalizado).

        Returns:
            Tupla (urgência, confiança).

        Raises:
            RuntimeError: Se o modelo não foi carregado.
        """
        if self._model is None:
            raise RuntimeError("Modelo não carregado. Chame load() primeiro.")
        probabilities = self._model.predict_proba([text])[0]
        class_index = apply_thresholds(
            probabilities.reshape(1, -1), self._thresholds, list(self._model.classes_)
        )[0]
        raw_label = int(self._model.classes_[class_index])
        confidence = min(1.0, max(0.0, float(probabilities[class_index])))
        urgency = URGENCY_MAP[raw_label]
        return urgency, confidence

    def predict_batch(self, texts: list[str]) -> list[tuple[UrgencyLevel, float]]:
        """Realiza inferência em lote com vetorização nativa (uma chamada ao modelo).

        Args:
            texts: Lista de textos de laudos médicos (normalizados).

        Returns:
            Lista de tuplas (urgência, confiança).

        Raises:
            RuntimeError: Se o modelo não foi carregado.
        """
        if self._model is None:
            raise RuntimeError("Modelo não carregado. Chame load() primeiro.")
        probabilities = self._model.predict_proba(texts)
        indices = apply_thresholds(probabilities, self._thresholds, list(self._model.classes_))
        results: list[tuple[UrgencyLevel, float]] = []
        for i, idx in enumerate(indices):
            raw_label = int(self._model.classes_[idx])
            conf = min(1.0, max(0.0, float(probabilities[i][idx])))
            results.append((URGENCY_MAP[raw_label], conf))
        return results


@register_loader("onnx")
class OnnxLoader(ModelLoader):
    """Carregador de modelos no formato ONNX.

    Carrega o modelo ONNX via onnxruntime e realiza inferência
    com verificação de integridade SHA256.

    Opcionalmente carrega limiares de decisão de um arquivo sidecar
    para ajustar as predições e maximizar o recall macro.
    """

    def __init__(self) -> None:
        self._session = None
        self._input_name: str = ""
        self._thresholds: dict[str, float] = dict(DEFAULT_THRESHOLDS)

    def load(self, path: Path | str) -> None:
        """Carrega o modelo ONNX com verificação de integridade e validação de classes.

        Args:
            path: Caminho para o arquivo .onnx.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se o hash SHA256 não corresponder ou as classes forem inválidas.
        """
        import onnxruntime as ort

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de modelo ONNX não encontrado: {path}")
        _verify_model_hash(path)
        self._session = ort.InferenceSession(
            path.read_bytes(),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        _validate_onnx_outputs(self._session)
        self._thresholds = _load_thresholds_for_model(path)

    def predict(self, text: str) -> tuple[UrgencyLevel, float]:
        """Prediz o nível de urgência via ONNX Runtime.

        Args:
            text: Texto do laudo médico (normalizado).

        Returns:
            Tupla (urgência, confiança).

        Raises:
            RuntimeError: Se o modelo não foi carregado.
        """
        if self._session is None:
            raise RuntimeError("Modelo ONNX não carregado. Chame load() primeiro.")
        input_data = np.array([text]).reshape(-1, 1)
        outputs = self._session.run(None, {self._input_name: input_data})
        probas = outputs[1]
        indices = apply_thresholds(probas, self._thresholds, EXPECTED_CLASSES)
        raw_label = int(indices[0])
        confidence = min(1.0, max(0.0, float(probas[0].max())))
        urgency = URGENCY_MAP[raw_label]
        return urgency, confidence

    def predict_batch(self, texts: list[str]) -> list[tuple[UrgencyLevel, float]]:
        """Realiza inferência em lote via ONNX Runtime.

        Args:
            texts: Lista de textos de laudos médicos (normalizados).

        Returns:
            Lista de tuplas (urgência, confiança).

        Raises:
            RuntimeError: Se o modelo não foi carregado.
        """
        if self._session is None:
            raise RuntimeError("Modelo ONNX não carregado. Chame load() primeiro.")
        input_data = np.array(texts).reshape(-1, 1)
        outputs = self._session.run(None, {self._input_name: input_data})
        probas = outputs[1]
        indices = apply_thresholds(probas, self._thresholds, EXPECTED_CLASSES)
        results: list[tuple[UrgencyLevel, float]] = []
        for i, idx in enumerate(indices):
            raw_label = int(idx)
            confidence = min(1.0, max(0.0, float(probas[i].max())))
            urgency = URGENCY_MAP[raw_label]
            results.append((urgency, confidence))
        return results


def create_model(model_path: Path | str, backend: ModelBackend = "joblib") -> ModelLoader:
    """Fábrica de carregadores de modelo.

    Args:
        model_path: Caminho para o artefato de modelo.
        backend: Backend de carregamento ('joblib' ou 'onnx').

    Returns:
        Instância de ModelLoader com o modelo carregado.

    Raises:
        ValueError: Se o backend não for suportado.
    """
    if backend not in _LOADERS:
        raise ValueError(f"Backend '{backend}' não suportado. Opções: {list(_LOADERS.keys())}")
    loader = _LOADERS[backend]()
    loader.load(model_path)
    return loader


def _verify_model_hash(model_path: Path) -> None:
    """Verifica a integridade do modelo via hash SHA256.

    Compara o hash SHA256 do arquivo de modelo contra o hash salvo
    no arquivo .sha256 correspondente. Se o arquivo de hash não existir,
    a verificação é ignorada (modo degradado).

    Args:
        model_path: Caminho do arquivo de modelo.

    Raises:
        ValueError: Se o hash não corresponder.
    """
    hash_path = model_path.with_suffix(model_path.suffix + ".sha256")

    if not hash_path.exists():
        logger.warning(
            "Hash file %s not found — model integrity cannot be verified. "
            "This is acceptable in development but should not occur in production.",
            hash_path,
        )
        return

    expected_hash = hash_path.read_text(encoding="utf-8").strip()
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        expected_short = expected_hash[:16]
        actual_short = actual_hash[:16]
        raise ValueError(
            f"Integridade do modelo comprometida. "
            f"Hash esperado: {expected_short}... "
            f"Hash atual: {actual_short}..."
        )


def _validate_model_classes(model) -> None:
    """Valida que as classes do modelo correspondem ao mapeamento esperado.

    Args:
        model: Pipeline sklearn treinado.

    Raises:
        ValueError: Se as classes não correspondem ao esperado.
    """
    actual = [int(c) for c in model.classes_]
    if actual != EXPECTED_CLASSES:
        raise ValueError(
            f"Classes do modelo {actual} não correspondem ao esperado {EXPECTED_CLASSES}"
        )


def _validate_onnx_outputs(session) -> None:
    """Valida que as saídas do modelo ONNX cobrem as classes esperadas.

    Executa uma predição de sanidade e verifica se os rótulos produzidos
    estão no mapeamento URGENCY_MAP.

    Args:
        session: Sessão ONNX Runtime carregada.

    Raises:
        ValueError: Se algum rótulo ONNX estiver fora de {0, 1, 2}.
    """
    import numpy as np

    input_name = session.get_inputs()[0].name
    test_input = np.array([["sanity check"]])
    outputs = session.run(None, {input_name: test_input})
    labels = outputs[0]
    for label in labels.flat:
        if int(label) not in EXPECTED_CLASSES:
            raise ValueError(
                f"Rótulo ONNX {int(label)} fora das classes esperadas {EXPECTED_CLASSES}"
            )


def _load_thresholds_for_model(model_path: Path) -> dict[str, float]:
    """Carrega limiares de decisão do sidecar do modelo.

    Procura o arquivo de limiares em duas localizações:
    1. `models/thresholds.json` (padrão do pipeline)
    2. Sidecar ao lado do modelo: `<model_path>.thresholds.json`

    Se nenhum arquivo de limiares existir, retorna limiares neutros (0.0),
    permitindo que o sistema funcione sem ajuste de limiares.

    Args:
        model_path: Caminho do arquivo de modelo (.joblib ou .onnx).

    Returns:
        Dicionário com offsets por classe (chaves "0", "1", "2").
    """
    from src.core.dataset import THRESHOLDS_PATH

    candidates = [
        THRESHOLDS_PATH,
        model_path.with_suffix(model_path.suffix + ".thresholds.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_thresholds(candidate)
    return dict(DEFAULT_THRESHOLDS)
