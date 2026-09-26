"""LSH mediante la técnica de bandas (banding technique), sección 4 del
enunciado.

Divide cada firma MinHash de longitud m = b * r en b bandas de r filas.
Dos usuarios que coincidan exactamente en al menos una banda se declaran
"candidatos" (par probablemente similar). Esto evita el cálculo de todos
los C(n,2) pares exactos, enfocando el cómputo solo en los pares con alta
probabilidad de ser similares (ver curva S en minhash.s_curve).

Convención de índices: `sig` tiene forma (m, n_users); los candidatos son
tuplas de índices posicionales (i, j) con i < j, consistentes con las
filas/columnas de `jaccard.exact_jaccard_matrix`.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import comb
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

Pair = Tuple[int, int]


def lsh_candidates(sig: np.ndarray, b: int, r: int, verbose: bool = True) -> Set[Pair]:
    """Divide la firma en b bandas de r filas.

    Para cada banda, agrupa usuarios por su sub-firma (como tupla) en un
    diccionario (bucket). Pares en el mismo bucket son candidatos.

    Retorna
    -------
    candidates : set of tuples (i, j) con i < j
    """
    m, n_users = sig.shape
    candidates: Set[Pair] = set()

    for band in tqdm(range(b), desc=f"lsh_candidates (b={b}, r={r})"):
        start, end = band * r, band * r + r
        buckets: Dict[tuple, List[int]] = defaultdict(list)
        for j in range(n_users):
            sub_sig = tuple(sig[start:end, j])
            buckets[sub_sig].append(j)

        for bucket_users in buckets.values():
            if len(bucket_users) > 1:
                for i, j in combinations(sorted(bucket_users), 2):
                    candidates.add((i, j))

    if verbose:
        total_pairs = comb(n_users, 2)
        frac = len(candidates) / total_pairs if total_pairs else 0.0
        print(
            f"Pares candidatos: {len(candidates):,} / {total_pairs:,} "
            f"totales ({frac:.4%})"
        )

    return candidates


def evaluate_lsh(candidates: Set[Pair], sim_matrix: np.ndarray, threshold: float) -> dict:
    """Calcula precision y recall de LSH contra la verdad (sim_matrix >= threshold).

    Retorna
    -------
    dict con: true_similar, candidates, true_positives, precision, recall
    """
    n = sim_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    true_similar = int((sim_matrix[iu] >= threshold).sum())

    true_positives = 0
    for i, j in tqdm(candidates, desc="evaluate_lsh"):
        if sim_matrix[i, j] >= threshold:
            true_positives += 1

    n_candidates = len(candidates)
    precision = true_positives / n_candidates if n_candidates else 0.0
    recall = true_positives / true_similar if true_similar else 0.0

    return {
        "true_similar": true_similar,
        "candidates": n_candidates,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
    }


def candidate_similarities(candidates: Set[Pair], sim_matrix: np.ndarray) -> np.ndarray:
    """Devuelve el array de similitudes reales de los pares candidatos."""
    sims = np.empty(len(candidates), dtype=np.float64)
    for k, (i, j) in enumerate(candidates):
        sims[k] = sim_matrix[i, j]
    return sims
