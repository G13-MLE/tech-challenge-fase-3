"""Testes de exportação ONNX e paridade de predição (joblib vs ONNX)."""

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.core.dataset import atomic_write_bytes, atomic_write_text
from src.models.factory import OnnxLoader, create_model
from src.models.urgency import URGENCY_MAP

SEED = 42
N_ESTIMATORS = 10
MAX_FEATURES = 100

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


@pytest.fixture()
def mini_pipeline():
    """Pipeline TF-IDF + RandomForest treinado com dados sintéticos."""
    texts = [text for text, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES)),
            ("clf", RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED)),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


@pytest.fixture()
def mini_joblib_path(tmp_path, mini_pipeline):
    """Salva o modelo joblib e seu hash SHA256."""
    model_path = tmp_path / "model.joblib"
    joblib.dump(mini_pipeline, model_path)
    hash_path = tmp_path / "model.joblib.sha256"
    import hashlib

    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    atomic_write_text(hash_path, sha256)
    return model_path


@pytest.fixture()
def mini_onnx_path(tmp_path, mini_pipeline, mini_joblib_path):
    """Exporta o modelo para ONNX e salva com hash SHA256."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType

    initial_types = [("input", StringTensorType([None, 1]))]
    onnx_model = convert_sklearn(
        mini_pipeline,
        initial_types=initial_types,
        target_opset=15,
        options={id(mini_pipeline): {"zipmap": False}},
    )
    onnx_bytes = onnx_model.SerializeToString()
    onnx_path = tmp_path / "model.onnx"
    atomic_write_bytes(onnx_path, onnx_bytes)

    import hashlib

    sha256 = hashlib.sha256(onnx_bytes).hexdigest()
    atomic_write_text(tmp_path / "model.onnx.sha256", sha256)

    return onnx_path


class TestOnnxExport:
    """Testes de exportação ONNX."""

    def test_export_creates_onnx_file(self, tmp_path, mini_pipeline, mini_joblib_path):
        """Exportação ONNX deve criar arquivo .onnx válido."""
        from src.train.export_onnx import export_onnx

        onnx_path = export_onnx(
            model_path=mini_joblib_path,
            onnx_path=tmp_path / "model.onnx",
            onnx_hash_path=tmp_path / "model.onnx.sha256",
        )
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_export_creates_hash_file(self, tmp_path, mini_pipeline, mini_joblib_path):
        """Exportação ONNX deve criar arquivo .sha256."""
        from src.train.export_onnx import export_onnx

        onnx_path = tmp_path / "model.onnx"
        hash_path = tmp_path / "model.onnx.sha256"
        export_onnx(
            model_path=mini_joblib_path,
            onnx_path=onnx_path,
            onnx_hash_path=hash_path,
        )
        assert hash_path.exists()
        hash_content = hash_path.read_text().strip()
        assert len(hash_content) == 64  # SHA256 hex

    def test_parity_joblib_vs_onnx(self, mini_joblib_path, mini_onnx_path):
        """Predições joblib e ONNX devem ser consistentes."""
        loader = OnnxLoader()
        loader.load(mini_onnx_path)

        texts = [text for text, _ in TRAINING_DATA]
        for text in texts:
            urgency, confidence = loader.predict(text)
            assert urgency in URGENCY_MAP.values()
            assert 0.0 <= confidence <= 1.0 + 1e-6

    def test_onnx_loader_predict_batch(self, mini_onnx_path):
        """OnnxLoader.predict_batch deve retornar resultados para todos os textos."""
        loader = OnnxLoader()
        loader.load(mini_onnx_path)

        texts = [text for text, _ in TRAINING_DATA]
        results = loader.predict_batch(texts)
        assert len(results) == len(texts)
        for urgency, confidence in results:
            assert urgency in URGENCY_MAP.values()
            assert 0.0 <= confidence <= 1.0 + 1e-6

    def test_onnx_loader_file_not_found(self, tmp_path):
        """OnnxLoader deve levantar FileNotFoundError se o arquivo não existe."""
        loader = OnnxLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.onnx")


class TestCreateModelOnnx:
    """Testa create_model com backend ONNX."""

    def test_create_model_onnx_backend(self, mini_onnx_path):
        """create_model com backend 'onnx' deve carregar e predizer corretamente."""
        model = create_model(mini_onnx_path, backend="onnx")
        urgency, confidence = model.predict("paciente com pneumonia grave")
        assert urgency in URGENCY_MAP.values()
        assert 0.0 <= confidence <= 1.0 + 1e-6
