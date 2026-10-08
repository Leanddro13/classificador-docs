from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DocumentoRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=5,
        description="Texto extraído do OCR ou PDF.",
    )
    nome_arquivo: Optional[str] = Field(
        default=None,
        description="Nome original do arquivo. Opcional e usado como feature auxiliar.",
    )

    @field_validator("texto")
    @classmethod
    def validar_texto(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("O texto precisa possuir pelo menos 5 caracteres.")
        return value


class ClassificacaoResponse(BaseModel):
    categoria: str
    confianca: float
    todas_probabilidades: dict[str, float]
    modelo: str


class FeedbackRequest(BaseModel):
    texto: str = Field(..., min_length=5)
    categoria_correta: str = Field(..., min_length=1)
    nome_arquivo: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    mensagem: str
    total_exemplos: int


class HealthResponse(BaseModel):
    status: str
    modelo_carregado: bool
    treinamento_em_andamento: bool
    classes_disponiveis: int
    versao_modelo: str
