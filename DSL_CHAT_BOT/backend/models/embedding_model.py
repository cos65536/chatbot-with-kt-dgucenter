from sentence_transformers import SentenceTransformer
from config.constants import EMBEDDING_MODEL

class EmbeddingModel:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        return self.embedder.encode(texts, convert_to_numpy=convert_to_numpy, show_progress_bar=show_progress_bar)

# 전역 인스턴스 (기존 호환성 유지)
embedding_instance = EmbeddingModel()