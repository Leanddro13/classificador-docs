from pydantic import BaseModel, Field

class DocumentoRequest(BaseModel):
    texto: str = Field(
        ..., min_lenght=10, description="Texto extraído do documento/OCR"
    )

class ClassificacaoResponse(BaseModel):
    categoria: str
    confianca: float
    todas_probabilidades: dict[str, float]

class FeedBackResquest(BaseModel):
    texto: str = Field(..., min_lenght=10)
    categoria_correta: str = Field(
        ..., description="Nome extato da pasta/categoria no Drive"
    )