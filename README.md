# Classificador HCM — Pipeline otimizado para OCR

A versão 2 substitui o pipeline `SentenceTransformer + LogisticRegression` por:

`texto OCR + nome_arquivo -> TF-IDF word (1,2) + TF-IDF char_wb (3,5) -> LinearSVC -> calibração de probabilidade`

O desenho foi escolhido para o dataset de documentos RH em português com ruído de OCR e baixa quantidade de exemplos em algumas classes.

## Treinamento

Execute no diretório raiz:

```bash
python train.py --dataset dataset_documentos_v4.csv --candidates 0.5,1,2
```

O treinamento:

1. Audita duplicatas e possíveis rótulos conflitantes.
2. Usa `StratifiedGroupKFold` com grupos por texto idêntico para reduzir leakage.
3. Seleciona `C` usando `f1_macro`.
4. Gera matriz de confusão e métricas em `data/results/`.
5. Treina o modelo final calibrado com o melhor `C`.
6. Faz substituição atômica do arquivo `.joblib`.

## API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Classificação

O `nome_arquivo` é opcional, mantendo compatibilidade com o contrato antigo.

```json
{
  "texto": "GUIA DE RECOLHIMENTO FGTS ...",
  "nome_arquivo": "Guia FGTS.pdf"
}
```

### Feedback

O feedback é armazenado no CSV e **não dispara retreino automático**. Isso evita degradação por dados individuais mal rotulados.

## Observação importante

Este pipeline não possui um limite artificial de 512 palavras. O TF-IDF funciona com documentos maiores. O ideal é remover o hard cap na etapa de OCR/extração. Para Transformers, chunking continua sendo necessário.
