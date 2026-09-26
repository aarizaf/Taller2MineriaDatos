"""Similitud de Jaccard exacta entre conjuntos de películas de usuarios.

Definición teórica (parcial_practico.pdf, sección 2 / lsh-slides.pdf):

    SIM(A, B) = |A ∩ B| / |A ∪ B|

Se implementa tanto el cálculo puntual (`jaccard`) como la construcción de
la matriz simétrica completa recorriendo todos los pares con
itertools.combinations (sección 2.2 del enunciado).

Convención de índices: `user_sets` es una lista de conjuntos donde la
posición `i` identifica al usuario `i` (0-indexado). Esta misma convención
de índices posicionales es la que usan `minhash.build_signatures`,
`lsh.lsh_candidates` y `lsh.evaluate_lsh`, de modo que `matrix[i, j]`,
`sig[:, i]` y los pares candidatos `(i, j)` siempre se refieren al mismo
usuario. El mapeo de vuelta a los userId reales de MovieLens se hace en el
notebook con la lista `users` (userId real en la posición i).
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import List, Set

import numpy as np
from tqdm import tqdm


def jaccard(set_a: Set[int], set_b: Set[int]) -> float:
    """Retorna la similitud de Jaccard entre dos conjuntos.

    SIM(A, B) = |A ∩ B| / |A ∪ B|

    Por convención, si ambos conjuntos son vacíos (unión = 0) la similitud
    se define como 0.0 para evitar división por cero.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def exact_jaccard_matrix(user_sets: List[Set[int]]) -> np.ndarray:
    """Calcula la similitud de Jaccard para todos los pares de usuarios.

    Recorre los C(n, 2) pares con itertools.combinations, almacena los
    resultados en una matriz simétrica y coloca 1.0 en la diagonal
    principal (cada usuario es idéntico a sí mismo).
    """
    n = len(user_sets)
    matrix = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)

    total_pairs = comb(n, 2)
    for i, j in tqdm(combinations(range(n), 2), total=total_pairs, desc="Jaccard exacto (pares)"):
        sim = jaccard(user_sets[i], user_sets[j])
        matrix[i, j] = sim
        matrix[j, i] = sim

    return matrix


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Extrae los valores del triángulo superior (sin la diagonal)."""
    n = matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    return matrix[iu]
