"""Paquete modular para el taller de Locality-Sensitive Hashing (LSH)
aplicado a similitud de usuarios sobre datasets MovieLens.

Módulos:
    data_loading  -> carga y exploración de datos (ratings.csv / ratings.dat)
    jaccard       -> similitud de Jaccard exacta
    minhash       -> firmas MinHash y estimación de similitud
    lsh           -> técnica de bandas (LSH) para búsqueda de candidatos
    viz           -> funciones de visualización reutilizables
"""

from . import data_loading, jaccard, minhash, lsh, viz

__all__ = ["data_loading", "jaccard", "minhash", "lsh", "viz"]
