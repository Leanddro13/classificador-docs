import logging
from pathlib import Path
import threading
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_PATH = DATA_DIR / 'modelo_pipeline.joblib'
DATASET_PATH = BASE_DIR / 'dataset_documentos.csv'

DATA_DIR.mkdir(parents=True, exist_ok=True)


class BertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, str):
            X = [X]
            
        return self.model.encode(X)


class DocumentClassifierService:
    def __init__(self):
        self.pipeline = None
        self.classes: list[str] = []
        self._lock = threading.Lock()
        self.is_training = False
        self.carregar_ou_treinar()

    def carregar_ou_treinar(self):
        with self._lock:
            if MODEL_PATH.exists():
                try:
                    self.pipeline = joblib.load(MODEL_PATH)
                    self.classes = list(self.pipeline.classes_)
                    logger.info(f"Modelo carregado com sucesso. Classes: {len(self.classes)}")
                    return
                except Exception as e:
                    logger.warning(f"Erro ao carregar modelo salvo ({e}). Forçando retreino...")

        self._disparar_treino_assincrono()

    def _disparar_treino_assincrono(self):
        """Inicia o treino em background para não travar a API do n8n"""
        if self.is_training:
            logger.info("Um treinamento já está em andamento. Ignorando requisição dupla.")
            return

        self.is_training = True
        thread = threading.Thread(target=self._processo_de_treino)
        thread.start()
        
    def forcar_retreino(self):
        """Método manual para ser engatilhado pela rota de webhook"""
        self._disparar_treino_assincrono()

    def _processo_de_treino(self):
        try:
            if not DATASET_PATH.exists():
                logger.error(f"Dataset não encontrado: {DATASET_PATH}")
                return

            logger.info("Iniciando leitura do dataset e treinamento do Pipeline Scikit-Learn...")
            df = pd.read_csv(DATASET_PATH).dropna(subset=['texto_limpo', 'categoria'])
            
            if len(df) == 0:
                logger.error("Dataset vazio após limpeza.")
                return

            X = df['texto_limpo'].tolist()
            y = df['categoria'].tolist()

            logger.info("Construindo e treinando o Pipeline...")
            pipeline = Pipeline([
                ('embedder', BertEmbedder('paraphrase-multilingual-MiniLM-L12-v2')),
                ('classifier', LogisticRegression(
                    C=5.0, 
                    class_weight='balanced', 
                    max_iter=2000, 
                    solver='lbfgs'
                ))
            ])

            logger.info(f'Treinando classificador com {len(df)} documentos...')
            pipeline.fit(X, y)
            
            logger.info("Salvando pipeline com joblib...")
            joblib.dump(pipeline, MODEL_PATH)

            with self._lock:
                self.pipeline = pipeline
                self.classes = list(self.pipeline.classes_)

            logger.info(f'Treinamento concluído com sucesso! Classes: {len(self.classes)}')

        except Exception as e:
            logger.error(f"Erro crítico durante o treinamento: {e}")
        finally: 
            self.is_training = False

    def classificar(self, texto: str) -> tuple[str, float, dict[str, float]]:
        # Se estiver no meio do primeiro treino e ainda não tiver modelo
        if not self.pipeline:
            return "TREINANDO_AGUARDE", 0.0, {}

        # Previsão
        probs = self.pipeline.predict_proba([texto])[0]
        idx_vencedor = int(np.argmax(probs))

        categoria = self.pipeline.classes_[idx_vencedor]
        confianca = float(probs[idx_vencedor])

        todas_probabilidades = {
            str(cls): round(float(prob), 4) for cls, prob in zip(self.pipeline.classes_, probs)
        }

        return categoria, round(confianca, 4), todas_probabilidades

    def adicionar_feedback(self, texto: str, categoria: str) -> int:
        """Adiciona o dado ao CSV sem engatilhar retreino automático"""
        
        # Tratar quebras de linha no texto
        texto_tratado = str(texto).replace('\n', ' ').replace('\r', ' ').strip()
        
        novo_registro = pd.DataFrame([{
            'nome_arquivo': 'feedback_manual_n8n.txt', 
            'categoria': str(categoria).strip(),
            'caminho_origem': 'via_webhook_n8n',
            'texto_limpo': texto_tratado
        }])

        with self._lock:
            header = not DATASET_PATH.exists()
            novo_registro.to_csv(
                DATASET_PATH,
                mode='a',
                header=header,
                index=False,
                encoding='utf-8',
            )
            # Lê só pra saber o tamanho atualizado
            total_linhas = sum(1 for _ in open(DATASET_PATH, encoding='utf-8')) - 1

        return total_linhas


classifier_service = DocumentClassifierService()