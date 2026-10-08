from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from app.config import CONFUSION_PATH, METRICS_PATH, MODEL_PATH
from app.model import (
    MODEL_VERSION,
    build_production_pipeline,
    build_raw_pipeline,
    prepare_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train")


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def atomic_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp, compress=3)
    tmp.replace(path)


def train(
    dataset: Path,
    c_candidates: Iterable[float],
    final_c: float | None,
) -> None:
    df, audit = prepare_dataset(dataset)
    logger.info("Auditoria: %s", json.dumps(audit, ensure_ascii=False))

    X = df[["texto_limpo", "nome_arquivo"]]
    y = df["categoria"]
    groups = df["group_id"]

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Busca orientada exatamente pela métrica objetivo do projeto.
    grid = GridSearchCV(
        estimator=build_raw_pipeline(c=1.0),
        param_grid={"classifier__C": list(c_candidates)},
        scoring="f1_macro",
        cv=cv,
        refit=True,
        n_jobs=1,
        return_train_score=True,
    )
    logger.info("Iniciando GridSearchCV (F1-Macro) ...")
    grid.fit(X, y, groups=groups)

    best_c = float(final_c if final_c is not None else grid.best_params_["classifier__C"])
    best_score = float(grid.best_score_)
    logger.info("Melhor C=%s | F1-Macro CV=%.4f", best_c, best_score)

    # Relatório out-of-fold do melhor C sem reusar amostras duplicadas entre folds.
    oof = pd.Series(index=df.index, dtype=object)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        model = build_raw_pipeline(best_c)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof.iloc[val_idx] = model.predict(X.iloc[val_idx])
        logger.info("Fold %d concluído", fold)

    mask = oof.notna()
    f1_oof = float(f1_score(y[mask], oof[mask], average="macro"))
    report = classification_report(y[mask], oof[mask], output_dict=True, zero_division=0)
    matrix = confusion_matrix(y[mask], oof[mask], labels=sorted(y.unique()))
    labels = sorted(y.unique())
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        CONFUSION_PATH,
        encoding="utf-8",
    )

    # Treino final com probabilidades calibradas.
    production = build_production_pipeline(best_c)
    production.fit(X, y)
    atomic_dump(production, MODEL_PATH)

    metrics = {
        "model_version": MODEL_VERSION,
        "dataset_path": str(dataset),
        "dataset_size": int(len(df)),
        "classes": int(y.nunique()),
        "best_C": best_c,
        "grid_best_f1_macro": best_score,
        "oof_f1_macro": f1_oof,
        "duplicate_groups": int(audit["grupos_textuais"]),
        "exact_duplicates": int(audit["duplicatas_exatas"]),
        "classes_distribution": audit["distribuicao_classes"],
        "classification_report_oof": report,
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Modelo salvo em %s", MODEL_PATH)
    logger.info("F1-Macro OOF = %.4f", f1_oof)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina o classificador híbrido de documentos RH.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="CSV de treinamento. Quando omitido, usa DATASET_PATH do projeto.",
    )
    parser.add_argument(
        "--candidates",
        type=parse_float_list,
        default=[0.5, 1.0, 2.0],
        help="Valores de C separados por vírgula.",
    )
    parser.add_argument(
        "--final-c",
        type=float,
        default=None,
        help="Força C final. Útil em retreino automatizado.",
    )
    args = parser.parse_args()

    from app.config import DATASET_PATH

    dataset_path = args.dataset or DATASET_PATH
    train(
        dataset=dataset_path,
        c_candidates=args.candidates,
        final_c=args.final_c,
    )
