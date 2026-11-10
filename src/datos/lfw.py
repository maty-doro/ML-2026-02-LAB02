"""Carga de LFW. La selección de identidades la completa el alumno."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people
from src.config import DIR_CACHE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros


class CargadorLFW(CargadorRostros):
    """Fotografías de LFW con mayor diversidad de pose, luz y fondo.

    TODO(alumno): implementar ``cargar`` y devolver un ``ConjuntoRostros``.
    Cuando este método deje de lanzar ``EtapaPendienteAlumno``,
    ``python main.py lfw`` reutiliza el mismo bucle de experimentos que Olivetti.

    Pasos esperados:

    1. ``fetch_lfw_people(min_faces_per_person=20, resize=0.5, data_home=...)``.
    2. Conservar como máximo 50 identidades, las de mayor frecuencia.
    3. Reindexar ``y`` con ``np.unique(..., return_inverse=True)`` y guardar los nombres.
    4. Leer alto y ancho desde ``images.shape``.
    5. Llamar a ``validar_conjunto`` antes de retornar.
    """

    nombre = "lfw"

    def cargar(self) -> ConjuntoRostros:

      lfw_data = fetch_lfw_people(
        min_faces_per_person = 20,
        resize = 0.5,
        data_home=str(DIR_CACHE),
        color=False
      )

      images = lfw_data.images
      X = lfw_data.data
      y = lfw_data.target
      nombres_originales = lfw_data.target_names

      etiquetas_unicas, conteos = np.unique(y, return_counts=True)

      indices_top_50 = np.argsort(conteos)[::-1][:50]
      clases_a_conservar = etiquetas_unicas[indices_top_50]

      mascara = np.isin(y, clases_a_conservar)
      images = images[mascara]
      X = X[mascara]
      y_filtrado = y[mascara]

      y_nuevas_etiquetas, mapeo_inverso = np.unique(y_filtrado, return_inverse=True)
      nombres_identidades = [nombres_originales[idx] for idx in y_nuevas_etiquetas]

      alto, ancho = images.shape[1], images.shape[2]
     
      conjunto = ConjuntoRostros(
          nombre=self.nombre,
          images=images,
          X=X,
          y=mapeo_inverso,
          nombres_identidades=nombres_identidades,
          alto=alto,
          ancho=ancho,
      )

      validar_conjunto(conjunto)
      return conjunto
