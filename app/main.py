from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import MODEL_PATH
from app.model import MODEL_VERSION, classifier_service
from app.schemas import (
    ClassificacaoResponse,
    DocumentoRequest,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)

app = FastAPI(
    title="Classificador de Documentos ML",
    version="2.0.0",
    description=(
        "Classificador de documentos RH baseado em TF-IDF híbrido "
        "(palavras + caracteres) e LinearSVC com calibração de probabilidades."
    ),
)

# Em produção, prefira configurar CORS via variável de ambiente/reverse proxy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="online" if classifier_service.model_loaded else "degraded",
        modelo_carregado=classifier_service.model_loaded,
        treinamento_em_andamento=classifier_service.is_training,
        classes_disponiveis=len(classifier_service.classes),
        versao_modelo=MODEL_VERSION if MODEL_PATH.exists() else "sem_modelo",
    )


@app.post("/classificar", response_model=ClassificacaoResponse)
def classificar_documento(payload: DocumentoRequest) -> ClassificacaoResponse:
    try:
        cat, conf, probs = classifier_service.classificar(
            texto=payload.texto,
            nome_arquivo=payload.nome_arquivo,
        )
        return ClassificacaoResponse(
            categoria=cat,
            confianca=conf,
            todas_probabilidades=probs,
            modelo=MODEL_VERSION,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na classificação: {exc}",
        ) from exc


@app.post("/feedback", response_model=FeedbackResponse)
def receber_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    try:
        total = classifier_service.adicionar_feedback(
            texto=payload.texto,
            categoria=payload.categoria_correta,
            nome_arquivo=payload.nome_arquivo,
        )
        return FeedbackResponse(
            status="sucesso",
            mensagem=(
                f"Documento registrado sob a categoria "
                f"'{payload.categoria_correta}'. "
                "O treinamento não é disparado automaticamente."
            ),
            total_exemplos=total,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar feedback: {exc}",
        ) from exc


@app.post("/retreinar")
def forcar_retreino() -> dict[str, str]:
    try:
        classifier_service.forcar_retreino()
        return {
            "status": "sucesso",
            "mensagem": (
                "Retreino iniciado em background. "
                "O modelo atual continua ativo até o novo modelo ser aprovado."
            ),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao iniciar retreino: {exc}",
        ) from exc
