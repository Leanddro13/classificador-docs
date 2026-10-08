from pydantic import BaseModel, Field


class DocumentoRequest(BaseModel):
  texto: str = Field(
      ..., min_length=5, description="Texto extraído do OCR ou PDF"
  )


class ClassificacaoResponse(BaseModel):
  categoria: str
  confianca: float
  todas_probabilidades: dict[str, float]


class FeedbackRequest(BaseModel):
  texto: str = Field(..., min_length=5)
  categoria_correta: str


class FeedbackResponse(BaseModel):
  status: str
  mensagem: str
  total_exemplos: int