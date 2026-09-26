"""Funciones de visualización reutilizables para el taller de LSH.

Todas las funciones guardan la figura en disco (carpeta reports/figures)
y también la retornan/mustran, para poder incrustarlas en el notebook.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def plot_hist_with_percentiles(
    data: np.ndarray,
    percentiles: Sequence[int] = (50, 90, 95),
    bins: int = 50,
    title: str = "",
    xlabel: str = "",
    filename: str = "hist_percentiles.png",
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    colors = ["red", "green", "purple", "orange", "brown", "black"]
    for p, color in zip(percentiles, colors):
        val = np.percentile(data, p)
        ax.axvline(val, color=color, linestyle="--", linewidth=1.5, label=f"P{p} = {val:.2f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.legend()
    fig.tight_layout()
    path = _save(fig, filename)
    plt.show()
    return path


def plot_error_curves(
    real_sim: np.ndarray,
    abs_err: np.ndarray,
    rel_err_pct: np.ndarray,
    theoretical_error: float,
    filename: str = "minhash_errors.png",
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(real_sim, abs_err, s=6, alpha=0.4, color="teal")
    axes[0].axhline(theoretical_error, color="red", linestyle="--",
                     label=f"Error teórico 1/√m = {theoretical_error:.4f}")
    axes[0].set_xlabel("Similitud real (Jaccard)")
    axes[0].set_ylabel("Error absoluto")
    axes[0].set_title("Error absoluto MinHash vs Jaccard real")
    axes[0].legend()

    axes[1].scatter(real_sim, rel_err_pct, s=6, alpha=0.4, color="darkorange")
    axes[1].set_xlabel("Similitud real (Jaccard)")
    axes[1].set_ylabel("Error relativo (%)")
    axes[1].set_title("Error relativo porcentual MinHash vs Jaccard real")

    fig.tight_layout()
    path = _save(fig, filename)
    plt.show()
    return path


def plot_s_curve(
    r_values: Iterable[int],
    t: float,
    lsh_params_fn,
    filename: str = "s_curve.png",
) -> str:
    from .minhash import s_curve

    s = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in r_values:
        b, m = lsh_params_fn(t, r)
        p = s_curve(s, b, r)
        ax.plot(s, p, label=f"r={r} (b={b}, m={m})")
    ax.axvline(t, color="black", linestyle="--", label=f"umbral t = {t:.3f}")
    ax.set_xlabel("Similitud real s")
    ax.set_ylabel("P(s) = 1 - (1 - s^r)^b")
    ax.set_title("Curva S: probabilidad de ser declarado candidato")
    ax.legend()
    fig.tight_layout()
    path = _save(fig, filename)
    plt.show()
    return path


def plot_candidates_vs_all(
    all_sims: np.ndarray,
    candidate_sims: np.ndarray,
    t: float,
    filename: str = "candidates_vs_all.png",
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 60)
    ax.hist(all_sims, bins=bins, alpha=0.5, label="Todos los pares", color="gray", density=True)
    ax.hist(candidate_sims, bins=bins, alpha=0.6, label="Pares candidatos (LSH)", color="crimson", density=True)
    ax.axvline(t, color="black", linestyle="--", label=f"umbral t = {t:.3f}")
    ax.set_xlabel("Similitud de Jaccard real")
    ax.set_ylabel("Densidad")
    ax.set_title("Similitud real: candidatos LSH vs. todos los pares")
    ax.legend()
    fig.tight_layout()
    path = _save(fig, filename)
    plt.show()
    return path
