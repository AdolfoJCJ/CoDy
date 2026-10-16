# CoDy — A Computational Framework for Representing
# the Temporal Organization of Conversation
#
# Copyright (C) 2026 Adolfo J. Céspedes
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License,
# version 3, as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Affero General Public License for more details.


# ============================
# app.py - CoDy continuous multivariate analysis application
# ============================

import numpy as np

from .config import GEOM_COMPACTNESS_WINDOW

def compute_ic3_geometry(
    E: np.ndarray,
    eps: float = 1e-12,
) -> dict:
    """
    Causal intrinsic geometry of the embedding trajectory.

    d_i(t): angular displacement between consecutive unit-normalized turn
    embeddings, divided by pi so that d_i in [0,1].

    kappa_i(t): turning angle between consecutive displacement vectors
    v_{t-1}=E_{t-1}-E_{t-2} and v_t=E_t-E_{t-1}, also divided by pi.
    It measures change in trajectory direction rather than change in step size.

    Every value at t depends only on turns <= t. No dialogue-wide rescaling,
    future-derived threshold, or centered smoother is used.
    """
    E = np.asarray(E, float)
    n = int(E.shape[0]) if E.ndim == 2 else 0
    if n == 0:
        return {"d_i": np.zeros(0, float), "kappa_i": np.zeros(0, float)}

    En = E.copy()
    norms = np.linalg.norm(En, axis=1, keepdims=True) + eps
    En = En / norms

    d_i = np.zeros(n, float)
    step_vecs = np.zeros_like(En)
    step_norms = np.zeros(n, float)

    for t in range(1, n):
        cos_sim = float(np.dot(En[t], En[t - 1]))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        d_i[t] = float(np.arccos(cos_sim) / np.pi)

        v = En[t] - En[t - 1]
        step_vecs[t] = v
        step_norms[t] = float(np.linalg.norm(v))

    kappa_i = np.zeros(n, float)
    for t in range(2, n):
        a = step_vecs[t - 1]
        b = step_vecs[t]
        na = step_norms[t - 1]
        nb = step_norms[t]
        if na <= eps or nb <= eps:
            continue
        cos_turn = float(np.dot(a, b) / (na * nb + eps))
        cos_turn = float(np.clip(cos_turn, -1.0, 1.0))
        kappa_i[t] = float(np.arccos(cos_turn) / np.pi)

    return {
        "d_i": np.clip(d_i, 0.0, 1.0),
        "kappa_i": np.clip(kappa_i, 0.0, 1.0),
    }

def semantic_compactness_rho(
    E: np.ndarray,
    texts: list,
    w: int = 2,
    mode: str = "centroid",
    min_tokens: int = 0,
) -> np.ndarray:
    """
    Causal local semantic compactness rho_t in [0,1].

    At t the window is [t-w,...,t]. Embeddings are unit-normalized inside this
    function. Short turns are retained by default because brief contributions
    can be interactionally important. u_t=1-rho_t is the primary dispersion
    coordinate used by the Research Tutorial analysis.
    """
    E = np.asarray(E, float)
    n = int(E.shape[0]) if E.ndim == 2 else 0
    if n == 0:
        return np.zeros(0, float)

    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    En = E / norms

    tok_ok = np.ones(n, dtype=bool)
    if texts is not None and min_tokens is not None and int(min_tokens) > 0:
        for i, t in enumerate(texts[:n]):
            s = t if isinstance(t, str) else ""
            tok_ok[i] = len(s.strip().split()) >= int(min_tokens)

    rho = np.zeros(n, float)
    w = int(max(0, w))
    for t in range(n):
        start = max(0, t - w)
        idx = np.arange(start, t + 1)
        idx = idx[tok_ok[idx]]
        if idx.size < 2:
            rho[t] = 1.0
            continue

        window_vecs = En[idx]
        if str(mode).lower().strip() == "pairwise":
            dists = []
            for i in range(len(window_vecs)):
                for j in range(i + 1, len(window_vecs)):
                    dists.append(float(np.linalg.norm(window_vecs[i] - window_vecs[j])))
            mean_d = float(np.mean(dists)) if dists else 0.0
        else:
            centroid = window_vecs.mean(axis=0)
            dists = np.linalg.norm(window_vecs - centroid, axis=1)
            mean_d = float(dists.mean())
        rho[t] = 1.0 / (1.0 + mean_d)

    return np.clip(rho, 0.0, 1.0)

def compute_all_signals(E, texts, Ct_base, C_inv, ic3):
    """Assemble the five primary CoDy observables without legacy composites."""
    rho_t = semantic_compactness_rho(
        E,
        texts,
        w=GEOM_COMPACTNESS_WINDOW,
        mode="centroid",
        min_tokens=0,
    )
    d_i = np.clip(np.asarray(ic3["d_i"], float), 0.0, 1.0)
    kappa_i = np.clip(np.asarray(ic3["kappa_i"], float), 0.0, 1.0)
    u_t = 1.0 - np.clip(rho_t, 0.0, 1.0)

    return {
        "Ct": np.asarray(Ct_base, float),
        "C_inv": np.asarray(C_inv, float) if C_inv is not None else None,
        "rho_t": np.asarray(rho_t, float),
        "u_t": np.asarray(u_t, float),
        "d_i": d_i,
        "kappa_i": kappa_i,
    }
