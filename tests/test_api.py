"""Testes da API de triagem e do factory de modelos."""

import pytest
from fastapi.testclient import TestClient
from starlette import status

from src.api.app import HEALTH_OK, app
from src.api.schemas import PredictRequest, PredictResponse
from src.models.factory import JoblibLoader, create_model
from src.models.urgency import UrgencyLevel


class TestSchemas:
    """Testes dos schemas Pydantic."""

    def test_predict_request_valid(self) -> None:
        request = PredictRequest(text="paciente com pneumonia")
        assert request.text == "paciente com pneumonia"

    def test_predict_request_empty_text_raises(self) -> None:
        with pytest.raises(ValueError):
            PredictRequest(text="")

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

    def test_load_and_predict(self) -> None:
        loader = JoblibLoader()
        loader.load("models/model.joblib")
        urgency, confidence = loader.predict("paciente com pneumonia grave")
        assert urgency in {UrgencyLevel.NORMAL, UrgencyLevel.ATENCAO, UrgencyLevel.URGENTE}
        assert 0.0 <= confidence <= 1.0

    def test_predict_without_load_raises(self) -> None:
        loader = JoblibLoader()
        with pytest.raises(RuntimeError, match="não carregado"):
            loader.predict("texto qualquer")

    def test_urgency_mapping_pneumonia(self) -> None:
        loader = JoblibLoader()
        loader.load("models/model.joblib")
        urgency, _ = loader.predict("paciente com pneumonia bacteriana grave")
        assert urgency == UrgencyLevel.URGENTE

    def test_urgency_mapping_diabetes(self) -> None:
        loader = JoblibLoader()
        loader.load("models/model.joblib")
        urgency, _ = loader.predict("paciente diabético com glicemia elevada")
        assert urgency == UrgencyLevel.ATENCAO

    def test_urgency_mapping_asma(self) -> None:
        loader = JoblibLoader()
        loader.load("models/model.joblib")
        urgency, _ = loader.predict("paciente com asma brônquica")
        assert urgency == UrgencyLevel.NORMAL


class TestCreateModel:
    """Testes da factory function."""

    def test_create_joblib_backend(self) -> None:
        model = create_model("models/model.joblib", backend="joblib")
        urgency, confidence = model.predict("hérnia inguinal")
        assert urgency in {UrgencyLevel.NORMAL, UrgencyLevel.ATENCAO, UrgencyLevel.URGENTE}
        assert 0.0 <= confidence <= 1.0

    def test_create_onnx_backend_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            create_model("models/model.onnx", backend="onnx")

    def test_create_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="não suportado"):
            create_model("models/model.joblib", backend="invalid")  # type: ignore[arg-type]


class TestAPI:
    """Testes dos endpoints da API."""

    def test_health(self) -> None:
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == status.HTTP_200_OK
            assert response.json() == {"status": HEALTH_OK}

    def test_predict_pneumonia(self) -> None:
        with TestClient(app) as client:
            response = client.post("/predict", json={"text": "paciente com pneumonia grave"})
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["urgency"] == UrgencyLevel.URGENTE.value
            assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_diabetes(self) -> None:
        with TestClient(app) as client:
            response = client.post("/predict", json={"text": "diabetes tipo 2 descompensada"})
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["urgency"] == UrgencyLevel.ATENCAO.value

    def test_predict_asma(self) -> None:
        with TestClient(app) as client:
            response = client.post("/predict", json={"text": "asma crônica com sibilos"})
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["urgency"] == UrgencyLevel.NORMAL.value

    def test_predict_empty_text_returns_422(self) -> None:
        with TestClient(app) as client:
            response = client.post("/predict", json={"text": ""})
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_missing_field_returns_422(self) -> None:
        with TestClient(app) as client:
            response = client.post("/predict", json={})
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
