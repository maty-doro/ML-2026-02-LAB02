from __future__ import annotations

import numpy as np

from src.embeddings.base import ExtractorEmbedding
from src.embeddings.pca import ExtractorPCA
from src.embeddings.lda import ExtractorLDA

class ExtractorFisherfaces(ExtractorEmbedding):
    nombre = "fisherfaces"

    def __init__(self, d_pca: int, d_lda: int) -> None:
        if d_pca < 1 or d_lda < 1:
            raise ValueError("Fisherfaces requiere d_pca y d_lda positivos.")
        if d_lda > d_pca:
            raise ValueError(
                f"d_lda={d_lda} supera d_pca={d_pca}. "
                "Debe cumplirse d_lda <= min(C-1, d_pca)."
            )
        self.d_pca = int(d_pca)
        self.d_lda = int(d_lda)

    @property
    def dimension(self) -> int:
        return self.d_lda

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorFisherfaces:
        self._pca = ExtractorPCA(n_components=self.d_pca)
        self._pca.fit(X)
        
        X_pca = self._pca.transform(X)
        
        self._lda = ExtractorLDA(n_components=self.d_lda)
        self._lda.fit(X_pca, y)
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_pca = self._pca.transform(X)
        z_final = self._lda.transform(X_pca)
        
        return self.comprobar(z_final, X.shape[0])
