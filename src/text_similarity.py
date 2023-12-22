import gc
from typing import List, Optional

import numpy as np
import seaborn as sns
import torch
from angle_emb import AnglE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import batch


class TextSimilarity:
    def __init__(self, embedding_model_path) -> None:
        self.embedding_model = AnglE.from_pretrained(
            embedding_model_path, pooling_strategy="cls"
        )

    def generate_embeddings(self, sentences: List[str], batch_size=1024) -> np.ndarray:
        try:
            embedding_model = self.embedding_model.cuda()
            batched_vectors = [
                embedding_model.encode(batched_sentences, to_numpy=True)
                for batched_sentences in batch(sentences, batch_size=batch_size)
            ]
            del embedding_model
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        vectors = np.concatenate(batched_vectors)

        return vectors

    def compute_similarity_matrix(
        self,
        queries: List[str],
        documents: Optional[List[str]] = None,
        documents_embeddings: Optional[np.ndarray] = None,
    ):
        if (documents is None and documents_embeddings is None) or (
            documents is not None and documents_embeddings is not None
        ):
            raise ValueError(
                "should have documents or documents_embeddings, but not both"
            )

        queries_embeddings = self.generate_embeddings(queries)
        if documents_embeddings is None:
            documents_embeddings = self.generate_embeddings(documents)

        similarity_matrix = cosine_similarity(queries_embeddings, documents_embeddings)
        return similarity_matrix

    def compute_pairwise_similarity_matrix(
        self,
        sentences: Optional[List[str]] = None,
        embedding_vectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if (sentences is None and embedding_vectors is None) or (
            sentences is not None and embedding_vectors is not None
        ):
            raise ValueError("should have sentences or embedding_vectors")

        if embedding_vectors is None:
            embedding_vectors = self.generate_embeddings(sentences)

        return cosine_similarity(embedding_vectors)

    def similarity(self, sentenceA: str, sentenceB: str) -> float:
        embedding_vectors = self.generate_embeddings((sentenceA, sentenceB))
        embedding_vectors = np.expand_dims(embedding_vectors, 1)
        return cosine_similarity(*embedding_vectors).item()

    def retrieve_n_most_similar(
        self,
        queries: List[str],
        documents: List[str],
        documents_embeddings: Optional[np.ndarray] = None,
        n=10,
    ):
        similarity_matrix = self.compute_similarity_matrix(
            queries,
            documents=documents if documents_embeddings is None else None,
            documents_embeddings=documents_embeddings,
        )
        indices = np.argsort(-similarity_matrix, axis=-1)[:, :n]
        return {
            query: (documents[idx], similarity_matrix[i, idx])
            for i, query in enumerate(queries)
            for idx in indices[i]
        }


def show_embedding_space(embedding_vectors, pca_components=0):
    if pca_components > 0:
        pca = PCA(n_components=pca_components)
        embedding_vectors = pca.fit_transform(embedding_vectors)
    tsne = TSNE(n_components=2)
    low_dim_emb = tsne.fit_transform(embedding_vectors)
    sns.scatterplot(x=low_dim_emb[:, 0], y=low_dim_emb[:, 1])
