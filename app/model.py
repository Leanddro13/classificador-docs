from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.config import (
    AUDIT_PATH,
    CALIBRATION_CV,
    CHAR_MAX_DF,
    CHAR_MIN_DF,
    CHAR_NGRAM_RANGE,
    CLASS_WEIGHT,
    CONFUSION_PATH,
    DATASET_PATH,
    MAX_FEATURES,
    MAX_INPUT_CHARS,
    METRICS_PATH,
    MIN_TEXT_CHARS,
    MIN_VALIDATION_F1,
    MODEL_PATH,
    SVM_C,
    WORD_MAX_DF,
    WORD_MIN_DF,
    WORD_NGRAM_RANGE,
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "2.0.0-hybrid-tfidf-svm"


def normalize_text(text: Any) -> str:
    """Normalização conservadora: reduz ruído estrutural sem destruir sinais de OCR."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_group_id(text: str) -> str:
    """Agrupa cópias textuais idênticas durante a validação para evitar leakage."""
    normalized = normalize_text(text).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def build_features() -> ColumnTransformer:
    """TF-IDF de palavras + caracteres, com o filename como feature auxiliar."""
    return ColumnTransformer(
        transformers=[
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=WORD_NGRAM_RANGE,
                    min_df=WORD_MIN_DF,
                    max_df=WORD_MAX_DF,
                    max_features=MAX_FEATURES,
                    sublinear_tf=True,
                    norm="l2",
                ),
                "texto_limpo",
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=CHAR_NGRAM_RANGE,
                    min_df=CHAR_MIN_DF,
                    max_df=CHAR_MAX_DF,
                    max_features=MAX_FEATURES,
                    sublinear_tf=True,
                    norm="l2",
                ),
                "texto_limpo",
            ),
            (
                "filename",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=CHAR_NGRAM_RANGE,
                    min_df=1,
                    sublinear_tf=True,
                    norm="l2",
                ),
                "nome_arquivo",
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def build_classifier(c: float = SVM_C) -> LinearSVC:
    return LinearSVC(
        C=c,
        class_weight=CLASS_WEIGHT,
        max_iter=5000,
        random_state=42,
    )


def build_raw_pipeline(c: float = SVM_C) -> Pipeline:
    """Pipeline usada para validação e seleção de hiperparâmetros."""
    return Pipeline(
        [
            ("features", build_features()),
            ("classifier", build_classifier(c)),
        ]
    )


def build_production_pipeline(c: float = SVM_C) -> Pipeline:
    """Pipeline final com probabilidades calibradas para a API."""
    calibrated = CalibratedClassifierCV(
        estimator=build_classifier(c),
        method="sigmoid",
        cv=CALIBRATION_CV,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("features", build_features()),
            ("classifier", calibrated),
        ]
    )


def prepare_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    df = pd.read_csv(path)
    original_rows = len(df)
    required = {"categoria", "texto_limpo"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    if "nome_arquivo" not in df.columns:
        df["nome_arquivo"] = ""

    if "caminho_origem" not in df.columns:
        df["caminho_origem"] = ""

    df["texto_limpo"] = df["texto_limpo"].fillna("").map(normalize_text)
    df["nome_arquivo"] = df["nome_arquivo"].fillna("").map(normalize_text)
    df["categoria"] = df["categoria"].fillna("").astype(str).str.strip()

    df = df[df["categoria"] != ""].copy()
    df = df[
        (df["texto_limpo"].str.len() >= MIN_TEXT_CHARS)
        | (df["nome_arquivo"].str.len() >= MIN_TEXT_CHARS)
    ].copy()

    # Texto vazio não deve formar um único grupo global: nesses casos o filename é a
    # única evidência disponível e os documentos podem pertencer a classes distintas.
    non_empty_text = df["texto_limpo"].str.len() >= MIN_TEXT_CHARS
    duplicate_groups = df.loc[non_empty_text].groupby("texto_limpo")["categoria"].nunique()
    conflicting = duplicate_groups[duplicate_groups > 1]
    if not conflicting.empty:
        sample = conflicting.index[:5].tolist()
        raise ValueError(
            "Foram encontradas duplicatas textuais com rótulos diferentes. "
            f"Audite os dados antes do treino. Exemplos: {sample}"
        )

    df["group_key"] = np.where(
        non_empty_text,
        df["texto_limpo"].str.lower(),
        "__EMPTY_TEXT__" + df["nome_arquivo"].str.lower(),
    )
    df["group_id"] = df["group_key"].map(make_group_id)
    df["qtd_palavras"] = df["texto_limpo"].str.split().str.len()

    audit = {
        "linhas_originais": int(original_rows),
        "linhas_validas": int(len(df)),
        "classes": int(df["categoria"].nunique()),
        "duplicatas_exatas": int(
            df.loc[non_empty_text, "texto_limpo"].duplicated().sum()
        ),
        "grupos_textuais": int(df["group_id"].nunique()),
        "textos_menos_10_palavras": int((df["qtd_palavras"] < 10).sum()),
        "textos_vazios": int((df["qtd_palavras"] == 0).sum()),
        "distribuicao_classes": {
            str(k): int(v) for k, v in df["categoria"].value_counts().items()
        },
    }
    return df, audit


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class DocumentClassifierService:
    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.classes: list[str] = []
        self._lock = threading.RLock()
        self.is_training = False
        self.last_training: dict[str, Any] = {}
        self.load()

    @property
    def model_loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        with self._lock:
            if not MODEL_PATH.exists():
                logger.warning(
                    "Modelo otimizado não encontrado em %s. "
                    "Execute train.py antes de colocar a API em produção.",
                    MODEL_PATH,
                )
                return

            try:
                pipeline = joblib.load(MODEL_PATH)
                if not isinstance(pipeline, Pipeline):
                    raise TypeError("Artefato salvo não é um sklearn Pipeline")
                classes = list(pipeline.classes_)
                self.pipeline = pipeline
                self.classes = classes
                logger.info(
                    "Modelo híbrido carregado: %s classes=%d",
                    MODEL_PATH,
                    len(classes),
                )
            except Exception:
                logger.exception("Falha ao carregar modelo %s", MODEL_PATH)
                self.pipeline = None
                self.classes = []

    def forcar_retreino(self) -> None:
        with self._lock:
            if self.is_training:
                logger.info("Treinamento já em andamento; requisição ignorada.")
                return
            self.is_training = True

        thread = threading.Thread(
            target=self._processo_de_treino,
            name="model-training",
            daemon=True,
        )
        thread.start()

    def _processo_de_treino(self) -> None:
        """Retreino operacional. A seleção de hiperparâmetros continua no train.py."""
        temp_path = MODEL_PATH.with_suffix(".joblib.tmp")
        try:
            df, audit = prepare_dataset(DATASET_PATH)
            save_json(AUDIT_PATH, audit)
            logger.info(
                "Dataset: %d documentos, %d classes, %d duplicatas exatas.",
                len(df),
                df["categoria"].nunique(),
                audit["duplicatas_exatas"],
            )

            # Gate mínimo: não substitui o modelo caso a validação básica falhe.
            # O modelo atualmente em produção continua intacto.
            from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

            X = df[["texto_limpo", "nome_arquivo"]]
            y = df["categoria"]
            groups = df["group_id"]
            cv = StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            )

            validation_scores = cross_val_score(
                build_raw_pipeline(SVM_C),
                X,
                y,
                groups=groups,
                cv=cv,
                scoring="f1_macro",
                n_jobs=1,
            )
            f1_macro = float(validation_scores.mean())
            f1_std = float(validation_scores.std())
            logger.info(
                "Validação 5-fold Group: F1-Macro=%.4f (+/- %.4f)",
                f1_macro,
                f1_std,
            )

            metrics = {
                "model_version": MODEL_VERSION,
                "f1_macro_mean": f1_macro,
                "f1_macro_std": f1_std,
                "fold_scores": [float(x) for x in validation_scores],
                "dataset_size": int(len(df)),
                "classes": int(df["categoria"].nunique()),
                "svm_C": SVM_C,
            }
            save_json(METRICS_PATH, metrics)

            if f1_macro < MIN_VALIDATION_F1:
                raise RuntimeError(
                    f"Quality gate reprovado: F1-Macro={f1_macro:.4f} < {MIN_VALIDATION_F1:.4f}"
                )

            X_train = df[["texto_limpo", "nome_arquivo"]]
            y_train = df["categoria"]
            pipeline = build_production_pipeline(SVM_C)
            pipeline.fit(X_train, y_train)

            temp_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, temp_path, compress=3)
            os.replace(temp_path, MODEL_PATH)

            with self._lock:
                self.pipeline = pipeline
                self.classes = list(pipeline.classes_)
                self.last_training = metrics

            logger.info(
                "Novo modelo salvo atomicamente em %s (%d classes).",
                MODEL_PATH,
                len(self.classes),
            )
        except Exception:
            logger.exception("Erro no retreino; modelo anterior foi preservado.")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        finally:
            with self._lock:
                self.is_training = False

    def classificar(
        self,
        texto: str,
        nome_arquivo: str | None = None,
    ) -> tuple[str, float, dict[str, float]]:
        with self._lock:
            pipeline = self.pipeline
            classes = list(self.classes)

        if pipeline is None:
            raise RuntimeError("Modelo ainda não está carregado.")

        texto = normalize_text(texto)
        nome_arquivo = normalize_text(nome_arquivo)
        if len(texto) > MAX_INPUT_CHARS:
            logger.warning(
                "Texto com %d caracteres excede o limite operacional de %d; truncando apenas na API.",
                len(texto),
                MAX_INPUT_CHARS,
            )
            texto = texto[:MAX_INPUT_CHARS]

        X = pd.DataFrame(
            [{"texto_limpo": texto, "nome_arquivo": nome_arquivo}]
        )
        probs = pipeline.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        categoria = classes[idx]
        confianca = float(probs[idx])
        todas = {
            str(cls): round(float(prob), 6)
            for cls, prob in zip(classes, probs)
        }
        return categoria, round(confianca, 6), todas

    def adicionar_feedback(
        self,
        texto: str,
        categoria: str,
        nome_arquivo: str | None = None,
    ) -> int:
        texto = normalize_text(texto)
        categoria = normalize_text(categoria)
        nome_arquivo = normalize_text(nome_arquivo)
        if len(texto) < MIN_TEXT_CHARS and len(nome_arquivo) < MIN_TEXT_CHARS:
            raise ValueError("Feedback sem texto/filename suficientes para treinamento.")

        # Mantém o schema compatível com o dataset atual.
        novo = pd.DataFrame(
            [
                {
                    "nome_arquivo": nome_arquivo or "feedback_manual_n8n.txt",
                    "categoria": categoria,
                    "caminho_origem": "via_webhook_n8n",
                    "texto_limpo": texto,
                }
            ]
        )

        with self._lock:
            header = not DATASET_PATH.exists()
            DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
            novo.to_csv(
                DATASET_PATH,
                mode="a",
                header=header,
                index=False,
                encoding="utf-8",
            )
            total = max(sum(1 for _ in DATASET_PATH.open(encoding="utf-8")) - 1, 0)

        return total


classifier_service = DocumentClassifierService()
