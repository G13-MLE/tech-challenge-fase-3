"""Factory Pattern para carregamento de modelos de ML.

Disponibiliza uma interface uniforme para carregar artefatos de modelo
a partir de diferentes backends (joblib, onnx), encapsulando a lógica
de desserialização e inferência.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Union

import joblib

from src.models.urgency import URGENCY_MAP, UrgencyLevel

ModelBackend = Literal["joblib", "onnx"]

_LOADERS: dict[str, type[ModelLoader]] = {}


class ModelLoader(ABC):
    """Interface abstrata para carregadores de modelo.

    Cada implementação define como um artefato de modelo é carregado
    e como a inferência é realizada sobre um texto de entrada.
    """

    @abstractmethod
    def load(self, path: Union[Path, str]) -> None:
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


class JoblibLoader(ModelLoader):
    """Carregador de modelos serializados com joblib.

    Espera um Pipeline sklearn com TfidfVectorizer e um classificador
    que implemente predict_proba. As classes do modelo devem mapear
    para os níveis de urgência na ordem: normal, atenção, urgente.
    """

    def __init__(self) -> None:
        self._model = None

    def load(self, path: Union[Path, str]) -> None:
        """Carrega o Pipeline sklearn do arquivo joblib.

        Args:
            path: Caminho para o arquivo .joblib.
        """
        path = Path(path)
        self._model = joblib.load(path)

    def predict(self, text: str) -> tuple[UrgencyLevel, float]:
        """Prediz o nível de urgência e a confiança associada.

        Args:
            text: Texto do laudo médico.

        Returns:
            Tupla (urgência, confiança).

        Raises:
            RuntimeError: Se o modelo não foi carregado.
        """
        if self._model is None:
            raise RuntimeError("Modelo não carregado. Chame load() primeiro.")
        probabilities = self._model.predict_proba([text])[0]
        predicted_class = probabilities.argmax()
        confidence = float(probabilities[predicted_class])
        urgency = URGENCY_MAP[predicted_class]
        return urgency, confidence


class OnnxLoader(ModelLoader):
    """Carregador de modelos no formato ONNX (stub).

    Reservado para integração futura na Etapa 4, quando o modelo
    poderá ser exportado para ONNX para inferência otimizada.
    """

    def load(self, path: Union[Path, str]) -> None:
        raise NotImplementedError("Backend ONNX será implementado na Etapa 4.")

    def predict(self, text: str) -> tuple[UrgencyLevel, float]:
        raise NotImplementedError("Backend ONNX será implementado na Etapa 4.")


_LOADERS["joblib"] = JoblibLoader
_LOADERS["onnx"] = OnnxLoader


def create_model(model_path: Union[Path, str], backend: ModelBackend = "joblib") -> ModelLoader:
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
