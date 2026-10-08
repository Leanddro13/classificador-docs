from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"

# Permite sobrescrever os caminhos no Docker/servidor sem alterar o código.
DATASET_PATH = Path(os.getenv("DATASET_PATH", BASE_DIR / "dataset_documentos.csv"))
if not DATASET_PATH.exists():
    fallback = BASE_DIR / "dataset_documentos_v4.csv"
    if fallback.exists():
        DATASET_PATH = fallback

# Modelo novo: não reutiliza acidentalmente o artefato antigo baseado em embeddings.
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", DATA_DIR / "modelo_hybrid_tfidf.joblib")
)
METRICS_PATH = RESULTS_DIR / "training_metrics.json"
AUDIT_PATH = RESULTS_DIR / "dataset_audit.json"
CONFUSION_PATH = RESULTS_DIR / "confusion_matrix.csv"

# Representação textual escolhida após o diagnóstico do dataset.
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)
WORD_MIN_DF = 2
CHAR_MIN_DF = 2
WORD_MAX_DF = 0.90
CHAR_MAX_DF = 0.95
MAX_FEATURES = 30000

# Melhor candidato observado na validação realizada sobre o dataset fornecido.
SVM_C = float(os.getenv("SVM_C", "1.0"))
CLASS_WEIGHT = "balanced"
CALIBRATION_CV = int(os.getenv("CALIBRATION_CV", "3"))

# Segurança operacional.
MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "5"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "500000"))
MIN_VALIDATION_F1 = float(os.getenv("MIN_VALIDATION_F1", "0.90"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
