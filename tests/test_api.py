"""Testes da API de triagem e do factory de modelos."""

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from starlette import status

from src.api.app import HEALTH_OK, app
from src.api.schemas import PredictRequest, PredictResponse
from src.models.factory import JoblibLoader, create_model
from src.models.urgency import UrgencyLevel

SEED = 42

TRAINING_DATA = [
    ("paciente com asma brônquica e sibilos", 0),
    ("asma crônica com sibilos", 0),
    ("crise asmática com broncoespasmo", 0),
    ("hérnia inguinal sem complicação", 0),
    ("hérnia umbilical redutível", 0),
    ("diabetes tipo 2 descompensada", 1),
    ("diabetes com hemoglobina glicada alta", 1),
    ("hipertensão arterial crônica", 1),
    ("crise hipertensiva com cefaleia", 1),
    ("pneumonia bacteriana grave", 2),
    ("pneumonia lobar com febre alta", 2),
    ("pneumonia comunitária com infiltrado", 2),
]


def build_mini_model(path) -> Pipeline:
    """Treina e salva um modelo mínimo em tmp_path para os testes."""
    texts = [text for text, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=100)),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=SEED)),
        ]
    )
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, path)
    return pipeline


class TestSchemas:
    """Testes dos schemas Pydantic."""

    def test_predict_request_valid(self) -> None:
        request = PredictRequest(text="paciente com pneumonia")
        assert request.text == "paciente com pneumonia"

    def test_predict_request_empty_text_raises(self) -> None:
        with pytest.raises(ValueError):
            PredictRequest(text="")

    def test_predict_request_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError):
            PredictRequest(text="   ")

    def test_predict_request_too_long_raises(self) -> None:
        with pytest.raises(ValueError):
            PredictRequest(text="a" * 10_001)

    def test_predict_request_strips_whitespace(self) -> None:
        request = PredictRequest(text="  paciente com pneumonia  ")
        assert request.text == "paciente com pneumonia"

    def test_predict_response_valid(self) -> None:
        response = PredictResponse(urgency=UrgencyLevel.URGENTE, confidence=0.95)
        assert response.urgency == UrgencyLevel.URGENTE
        assert response.confidence == 0.95

    def test_predict_response_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            PredictResponse(urgency=UrgencyLevel.NORMAL, confidence=1.5)

    def test_predict_response_negative_confidence(self) -> None:
        with pytest.raises(ValueError):
            PredictResponse(urgency=UrgencyLevel.NORMAL, confidence=-0.1)


class TestJoblibLoader:
    """Testes do carregador joblib."""

    def test_load_and_predict(self, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        build_mini_model(model_path)
        loader = JoblibLoader()
        loader.load(model_path)
        urgency, confidence = loader.predict("paciente com pneumonia grave")
        assert urgency in {UrgencyLevel.NORMAL, UrgencyLevel.ATENCAO, UrgencyLevel.URGENTE}
        assert 0.0 <= confidence <= 1.0

    def test_predict_without_load_raises(self) -> None:
        loader = JoblibLoader()
        with pytest.raises(RuntimeError, match="não carregado"):
            loader.predict("texto qualquer")

    def test_load_missing_file_raises(self, tmp_path) -> None:
        loader = JoblibLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "inexistente.joblib")

    def test_load_validates_classes(self, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        build_mini_model(model_path)
        loader = JoblibLoader()
        loader.load(model_path)
        assert list(loader._model.classes_) == [0, 1, 2]

    def test_urgency_mapping_pneumonia(self, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        build_mini_model(model_path)
        loader = JoblibLoader()
        loader.load(model_path)
        urgency, _ = loader.predict("paciente com pneumonia bacteriana grave")
        assert urgency == UrgencyLevel.URGENTE

    def test_urgency_mapping_diabetes(self, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        build_mini_model(model_path)
        loader = JoblibLoader()
        loader.load(model_path)
        urgency, _ = loader.predict("diabetes tipo 2 descompensada")
        assert urgency == UrgencyLevel.ATENCAO

    def test_urgency_mapping_asma(self, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        build_mini_model(model_path)
        loader = JoblibLoader()
        loader.load(model_path)
        urgency, _ = loader.predict("paciente com asma brônquica")
        assert urgency == UrgencyLevel.NORMAL


class TestCreateModel:
    """Testes da factory function."""

    def test_create_joblib_backend(self, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        build_mini_model(model_path)
        model = create_model(model_path, backend="joblib")
        urgency, confidence = model.predict("hérnia inguinal")
        assert urgency in {UrgencyLevel.NORMAL, UrgencyLevel.ATENCAO, UrgencyLevel.URGENTE}
        assert 0.0 <= confidence <= 1.0

    def test_create_onnx_backend_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            create_model("models/model.onnx", backend="onnx")

    def test_create_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="não suportado"):
            create_model("models/model.joblib", backend="invalid")  # type: ignore[arg-type]

    def test_predict_batch(self, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        build_mini_model(model_path)
        model = create_model(model_path, backend="joblib")
        results = model.predict_batch(["paciente com pneumonia", "asma brônquica"])
        assert len(results) == 2
        for urgency, confidence in results:
            assert urgency in {UrgencyLevel.NORMAL, UrgencyLevel.ATENCAO, UrgencyLevel.URGENTE}
            assert 0.0 <= confidence <= 1.0


class TestAPI:
    """Testes dos endpoints da API."""

    @pytest.fixture()
    def client_with_model(self, tmp_path, monkeypatch) -> TestClient:
        """Client com modelo treinado em tmp_path e MODEL_PATH injetado."""
        model_path = tmp_path / "model.joblib"
        build_mini_model(model_path)
        monkeypatch.setenv("MODEL_PATH", str(model_path))
        from src.core.config import get_settings

        get_settings.cache_clear()
        with TestClient(app) as client:
            yield client
        get_settings.cache_clear()

    def test_health(self, client_with_model) -> None:
        response = client_with_model.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": HEALTH_OK}

    def test_predict_pneumonia(self, client_with_model) -> None:
        response = client_with_model.post("/predict", json={"text": "paciente com pneumonia grave"})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["urgency"] == UrgencyLevel.URGENTE.value
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_diabetes(self, client_with_model) -> None:
        response = client_with_model.post(
            "/predict", json={"text": "diabetes tipo 2 descompensada"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["urgency"] == UrgencyLevel.ATENCAO.value

    def test_predict_asma(self, client_with_model) -> None:
        response = client_with_model.post("/predict", json={"text": "asma crônica com sibilos"})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["urgency"] == UrgencyLevel.NORMAL.value

    def test_predict_normalizes_input(self, client_with_model) -> None:
        response = client_with_model.post("/predict", json={"text": "PACIENTE, com PNEUMONIA!"})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["urgency"] == UrgencyLevel.URGENTE.value

    def test_predict_empty_text_returns_422(self, client_with_model) -> None:
        response = client_with_model.post("/predict", json={"text": ""})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_missing_field_returns_422(self, client_with_model) -> None:
        response = client_with_model.post("/predict", json={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
