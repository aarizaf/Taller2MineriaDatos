"""MinHash: compresión de conjuntos en firmas cortas y estimación de Jaccard.

Se sigue la teoría de lsh-slides.pdf y el enunciado (parcial_practico.pdf,
sección 3):
    - Familia de funciones hash h_k(x) = (a_k * x + b_k) mod p, que simulan
      permutaciones aleatorias del universo de ítems.
    - p es el primer número primo >= al universo de ítems N (sympy.nextprime).
    - La firma de un conjunto A bajo h_k es min_{x in A} h_k(x).
    - Propiedad fundamental de MinHash:
          Pr[min π(A) == min π(B)] = J(A, B)
      por lo que el estimador de similitud es la fracción de posiciones de
      la firma que coinciden entre dos usuarios.
    - Parámetros de bandas: b = ceil((1/t) ** r), m = b * r.

Convención de índices: al igual que en `jaccard.py`, `user_sets` es una
lista de conjuntos (posición i = usuario i), y la firma resultante `sig`
tiene forma (m, n_users), tal como pide el enunciado, de modo que
`sig[:, j]` es la firma completa del usuario j.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Set

import numpy as np
import pandas as pd
from sympy import nextprime
from tqdm import tqdm


def lsh_params(t: float, r: int) -> tuple[int, int]:
    """Calcula b y m a partir del umbral t y filas por banda r.

        b = ceil((1/t) ** r)
        m = b * r
    """
    b = math.ceil((1 / t) ** r)
    m = b * r
    return b, m


def lsh_params_table(t: float, r_values: Sequence[int]) -> pd.DataFrame:
    """Tabla de (r, b, m) para un umbral t y varios valores de r (sección 3.1)."""
    rows = [{"r": r, **dict(zip(("b", "m"), lsh_params(t, r)))} for r in r_values]
    return pd.DataFrame(rows)


def _item_universe(user_sets: List[Set[int]]) -> tuple[dict, int]:
    """Obtiene el universo de ítems e indexa cada ítem como entero [0, N-1]."""
    all_items: Set[int] = set()
    for s in user_sets:
        all_items |= s
    item_to_idx = {item: idx for idx, item in enumerate(sorted(all_items))}
    return item_to_idx, len(item_to_idx)


def build_signatures(user_sets: List[Set[int]], m: int, seed: int = 42) -> np.ndarray:
    """Construye la matriz de firmas MinHash.

    Parametros
    ----------
    user_sets : list of sets
    m : numero de funciones hash

    Retorna
    -------
    sig : np.ndarray de forma (m, n_users)
    """
    item_to_idx, n_items = _item_universe(user_sets)
    p = int(nextprime(n_items))

    rng = np.random.RandomState(seed)
    a = rng.randint(1, p, size=m)  # a_k en [1, p-1]
    b = rng.randint(0, p, size=m)  # b_k en [0, p-1]

    n_users = len(user_sets)
    sig = np.full((m, n_users), p, dtype=np.int64)

    for j, items in enumerate(tqdm(user_sets, desc=f"build_signatures (m={m})")):
        if not items:
            continue
        x = np.fromiter((item_to_idx[it] for it in items), dtype=np.int64, count=len(items))
        # h_k(x) = (a_k * x + b_k) mod p para todas las k y todos los x a la vez.
        # a: (m,), x: (n_items_usuario,) -> outer(a, x): (m, n_items_usuario)
        hashes = (np.outer(a, x) + b[:, None]) % p
        sig[:, j] = hashes.min(axis=1)

    return sig


def minhash_sim(sig: np.ndarray, i: int, j: int) -> float:
    """Similitud estimada = fraccion de filas donde sig[:,i] == sig[:,j]"""
    return np.mean(sig[:, i] == sig[:, j])


def build_signatures_v2(user_sets: List[Set[int]], m: int, seed: int = 42) -> np.ndarray:
    """Versión 2 (optimizada) de build_signatures para datasets muy grandes
    (p.ej. ml-32m). Ver justificación y benchmark en la sección 7 del notebook.

    Cuello de botella de build_signatures (v1): por cada aparición de un
    ítem x en el conjunto de CUALQUIER usuario, se recalculan las m
    evaluaciones h_k(x) = (a_k*x + b_k) mod p desde cero. Como un mismo
    ítem popular puede aparecer en miles de conjuntos de usuario, ese hash
    se recalcula miles de veces de forma redundante: el costo total es
    O(n_ratings * m), proporcional al número de RATINGS (32 millones en
    ml-32m), no al número de ítems distintos (~87,585 en ml-32m).

    Optimización: precalcular la tabla de hashes UNA sola vez para cada uno
    de los N ítems distintos del universo (tabla de forma (m, n_items)) y
    luego, para cada usuario, construir la firma indexando esa tabla y
    reduciendo con `min`. Esto baja el costo de la parte cara (evaluar h_k)
    de O(n_ratings * m) a O(n_items * m).
    """
    item_to_idx, n_items = _item_universe(user_sets)
    p = int(nextprime(n_items))

    rng = np.random.RandomState(seed)
    a = rng.randint(1, p, size=m)
    b = rng.randint(0, p, size=m)

    # Tabla de hashes de TODOS los ítems del universo, calculada una única vez.
    all_items = np.arange(n_items, dtype=np.int64)
    item_hash_table = (np.outer(a, all_items) + b[:, None]) % p  # shape (m, n_items)

    n_users = len(user_sets)
    sig = np.full((m, n_users), p, dtype=np.int64)

    for j, items in enumerate(tqdm(user_sets, desc=f"build_signatures_v2 (m={m})")):
        if not items:
            continue
        idx = np.fromiter((item_to_idx[it] for it in items), dtype=np.int64, count=len(items))
        sig[:, j] = item_hash_table[:, idx].min(axis=1)

    return sig


def s_curve(s: np.ndarray, b: int, r: int) -> np.ndarray:
    """Probabilidad de que la técnica de bandas declare candidato al par
    con similitud real s:  P(s) = 1 - (1 - s**r) ** b
    """
    return 1.0 - (1.0 - np.power(s, r)) ** b

