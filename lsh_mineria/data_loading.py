"""Carga y exploración de datos de ratings de MovieLens.

Contiene funciones para:
    - Cargar ratings.csv (ml-latest-small / ml-32m) y ratings.dat (ml-1m).
    - Calcular estadísticas descriptivas básicas del dataset.
    - Construir el conjunto de películas vistas por cada usuario.
    - Convertir la Serie userId -> set(movieId) a la representación
      `list of sets` que usan `minhash.build_signatures`,
      `jaccard.exact_jaccard_matrix` y `lsh.lsh_candidates`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

import pandas as pd
from tqdm import tqdm

tqdm.pandas()


@dataclass
class DatasetSummary:
    """Resumen descriptivo de un dataset de ratings."""

    n_ratings: int
    n_users: int
    n_movies: int

    def __str__(self) -> str:  # pragma: no cover - solo representación
        return (
            f"Ratings totales : {self.n_ratings:,}\n"
            f"Usuarios únicos : {self.n_users:,}\n"
            f"Películas únicas: {self.n_movies:,}"
        )


def load_ratings_csv(path: str) -> pd.DataFrame:
    """Carga ratings.csv (formato ml-latest-small / ml-25m / ml-32m).

    Columnas esperadas: userId, movieId, rating, timestamp.
    """
    df = pd.read_csv(path)
    return df


def load_ratings_dat(path: str) -> pd.DataFrame:
    """Carga ratings.dat (formato ml-1m), separador '::' con engine='python'."""
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        header=None,
        encoding="latin-1",
    )
    return df


def summarize_ratings(df: pd.DataFrame) -> DatasetSummary:
    """Calcula el total de ratings, usuarios únicos y películas únicas."""
    return DatasetSummary(
        n_ratings=len(df),
        n_users=df["userId"].nunique(),
        n_movies=df["movieId"].nunique(),
    )


def build_user_movie_sets(df: pd.DataFrame) -> pd.Series:
    """Construye el conjunto de películas vistas por cada usuario.

    Usa groupby + apply(set) tal como pide el enunciado. Se emplea
    `progress_apply` (tqdm.pandas) para monitorear el progreso.
    """
    user_sets = df.groupby("userId")["movieId"].progress_apply(set)
    return user_sets


def series_to_list(user_sets_series: pd.Series) -> Tuple[List[Set[int]], List[int]]:
    """Convierte la Serie (userId -> set(movieId)) a una lista de conjuntos.

    Retorna:
        user_sets : list of sets, donde la posición i es el usuario `users[i]`
        users     : lista de userId reales de MovieLens, en el mismo orden
                    que `user_sets` (y que las filas/columnas de la matriz
                    de Jaccard, las columnas de la firma MinHash, etc.)
    """
    users = list(user_sets_series.index)
    user_sets = [user_sets_series[u] for u in users]
    return user_sets, users

