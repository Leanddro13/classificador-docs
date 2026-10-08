from app.model import classifier_service
from app.schemas import (
    ClassificacaoResponse,
    DocumentoRequest,
    FeedbackRequest,
    FeedbackResponse,
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title='Classificador de Documentos ML',
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health_check():
  return {
      'status': 'online',
      'classes_disponiveis': len(classifier_service.classes),
  }


@app.post('/classificar', response_model=ClassificacaoResponse)
def classificar_documento(payload: DocumentoRequest):
  try:
    cat, conf, probs = classifier_service.classificar(payload.texto)
    return ClassificacaoResponse(
        categoria=cat, confianca=conf, todas_probabilidades=probs
    )
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f'Erro na classificação: {str(e)}'
    )


@app.post('/feedback', response_model=FeedbackResponse)
def receber_feedback(payload: FeedbackRequest):
  try:
    total = classifier_service.adicionar_feedback(
        texto=payload.texto, categoria=payload.categoria_correta
    )
    return FeedbackResponse(
        status='sucesso',
        mensagem=f"Documento registrado sob a categoria '{payload.categoria_correta}'. Treinamento não foi engatilhado automaticamente.",
        total_exemplos=total,
    )
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f'Erro ao processar feedback: {str(e)}'
    )


@app.post('/retreinar')
def forcar_retreino():
  try:
    classifier_service.forcar_retreino()
    return {
        'status': 'sucesso',
        'mensagem': 'Treinamento iniciado em background. Acompanhe os logs do servidor.'
    }
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f'Erro ao iniciar retreino: {str(e)}'
    )