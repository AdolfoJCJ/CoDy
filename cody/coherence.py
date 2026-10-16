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

from typing import List, Optional

import numpy as np
import pandas as pd

def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def _cosine_sim_matrix(E: np.ndarray) -> np.ndarray:
    """E: (n, d) embeddings. Returns cosine similarity matrix (n, n)."""
    E = np.asarray(E, dtype=float)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    En = E / norms
    return En @ En.T

def _build_weighted_adj_from_embeddings(
    E_window: np.ndarray,
    k_nn: int = 5,
    thr: float = 0.0,
    tie_eps: float = 1e-12,   
) -> np.ndarray:
    """
    Build symmetric weighted adjacency from cosine similarities.
    Uses k-NN sparsification.
    Deterministic tie-break avoids jitter when similarities are near-equal.
    """
    S = _cosine_sim_matrix(E_window)
    np.fill_diagonal(S, 0.0)
    S = np.clip(S, 0.0, 1.0)

    n = S.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    k = int(max(1, min(int(k_nn), n - 1)))
    A = np.zeros_like(S)

    # deterministic tie-break: add tiny increasing noise by column index
    col_bias = tie_eps * (np.arange(n, dtype=float) / max(1, n - 1))

    for i in range(n):
        row = S[i] + col_bias
        idx = np.argpartition(row, -k)[-k:]
        for j in idx:
            if S[i, j] >= thr:
                A[i, j] = S[i, j]

    A = np.maximum(A, A.T)
    return A

def _normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 0:
        return A
    d = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-12))
    I = np.eye(n)
    return I - D_inv_sqrt @ A @ D_inv_sqrt

def _laplacian_spectrum_features(A: np.ndarray, k_eigs: int = 6) -> np.ndarray:
    """
    Fixed-length feature vector from normalized Laplacian spectrum.
    Take smallest non-trivial eigenvalues (exclude ~0).
    """
    n = A.shape[0]
    k = int(max(1, k_eigs))

    if n < 3:
        return np.zeros((k,), dtype=float)

    L = _normalized_laplacian(A)
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(np.real(evals))

    evals_nt = evals[1:] if evals.size > 1 else evals
    take = evals_nt[:k]
    if take.size < k:
        take = np.pad(take, (0, k - take.size), constant_values=0.0)

    return np.clip(take.astype(float), 0.0, 2.0)

def _degree_stats(A: np.ndarray) -> np.ndarray:
    """Mean and std of weighted degree (2 dims)."""
    if A.size == 0:
        return np.zeros((2,), dtype=float)
    deg = A.sum(axis=1)
    return np.array([float(deg.mean()), float(deg.std())], dtype=float)

def compute_Pi_graph_invariants(
    E_window: np.ndarray,
    k_nn: int = 5,
    thr: float = 0.0,
    k_eigs: int = 6,
) -> np.ndarray:
    A = _build_weighted_adj_from_embeddings(E_window, k_nn=k_nn, thr=thr)
    spec = _laplacian_spectrum_features(A, k_eigs=k_eigs)
    degs = _degree_stats(A)
    return np.concatenate([spec, degs], axis=0)

def compute_C_inv_series(
    E: np.ndarray,
    window: int = 8,
    k_nn: int = 5,
    thr: float = 0.10,
    k_eigs: int = 6,
    D_max: Optional[float] = None,
) -> np.ndarray:
    """
    C_inv(t) compares Pi(G_t) vs Pi(G_{t-1}) for rolling window graphs.
    The first graph summary Pi(G_t) exists at t = window-1 (0-indexed), but
    C_inv itself needs two consecutive graph summaries. Therefore the first
    finite C_inv value can appear at t = window (turn window+1 if turns start at 1).
    Earlier values remain NaN so they cannot pollute scaling or downstream logic.
    """
    E = np.asarray(E, dtype=float)
    T = int(E.shape[0])
    if T == 0:
        return np.zeros((0,), dtype=float)

    W = int(max(3, window))
    Pi_list: List[Optional[np.ndarray]] = [None] * T
    Cinv = np.full((T,), np.nan, dtype=float)
    dists = np.full((T,), np.nan, dtype=float)

    # compute Pi only once window is full
    for t in range(T):
        if t < W - 1:
            continue
        Ewin = E[(t - W + 1):(t + 1)]
        Pi_list[t] = compute_Pi_graph_invariants(Ewin, k_nn=k_nn, thr=thr, k_eigs=k_eigs)

    # distances between consecutive valid Pi vectors
    for t in range(1, T):
        if Pi_list[t] is None or Pi_list[t - 1] is None:
            continue
        dists[t] = float(np.linalg.norm(Pi_list[t] - Pi_list[t - 1], ord=2))

    valid = np.isfinite(dists)
    if not np.any(valid):
        # if dialogue too short, return all-NaN (caller can handle)
        return Cinv

    if (D_max is None) or (not np.isfinite(D_max)) or (float(D_max) <= 0):
        scale = float(np.nanquantile(dists[valid], 0.95)) + 1e-12
    else:
        scale = float(D_max)

    Cinv[valid] = 1.0 - np.clip(dists[valid] / scale, 0.0, 1.0)
    return Cinv

def smooth_coherence(
    y: np.ndarray,
    method: str = "ema",
    ema_alpha: float = 0.20,
    ewma_span: int = 9,
) -> np.ndarray:
    y = np.asarray(y, float)
    if y.size == 0:
        return y
    method = (method or "ema").lower().strip()
    if method == "ewma":
        ewma_span = int(max(3, ewma_span))
        return pd.Series(y).ewm(span=ewma_span, adjust=False).mean().to_numpy()
    a = float(np.clip(ema_alpha, 0.0, 0.999))
    return _ema(y, alpha=a)

def _sigma(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return 1.0 / (1.0 + np.exp(-x))

def compute_ic2_dynamics(
    E: np.ndarray,
    alpha_context: float = 0.84,
    b: float = 0.40,
    eps: float = 1e-12,
) -> dict:
    """
Canonical IC-II contextual coherence.

1. Unit-normalize turn embeddings.

2. Construct a hybrid semantic context:
   global_ctx(t) = a * I_m(t-1) + (1-a) * E(t-1)
   local_ctx(t)  = mean of up to the previous 4 turn embeddings
   I_m(t)        = 0.3 * global_ctx(t) + 0.7 * local_ctx(t)

3. Measure:
   resonance(t)    = cosine(E(t), I_m(t))
   displacement(t) = ||E(t) - E(t-1)||

4. Construct a temporally persistent identity trajectory from
   resonance and normalized displacement, with a recovery term.

5. Convert the normalized identity trajectory into contextual
   coherence C_t using:
   C_t = sigmoid(3 * (identity_traj - b))
"""
    E = np.asarray(E, float)
    n = E.shape[0]
    if n == 0:
        z = np.zeros(0, float)
        return {"I_s": E, "I_m": E, "res": z, "dI_norm": z, "C_t": z}

    # unit-normalize embeddings for stable cosine + geometry consistency
    I_s = E.copy()
    norms = np.linalg.norm(I_s, axis=1, keepdims=True) + eps
    I_s = I_s / norms

    # --- Hybrid context (GLOBAL + LOCAL) ---
    I_m = np.zeros_like(I_s)

    I_m[0] = I_s[0]
    a = float(np.clip(alpha_context, 0.0, 0.999))
    window = 4  # 🔑 clave: 3–6

    for t in range(1, n):
        # GLOBAL (long memory)
        global_ctx = a * I_m[t - 1] + (1.0 - a) * I_s[t - 1]

        # LOCAL (last turns)
        start = max(0, t - window)
        local_ctx = np.mean(I_s[start:t], axis=0)

        # COMBINATION
        I_m[t] = 0.3 * global_ctx + 0.7 * local_ctx

    # resonance r_t
    res = np.zeros(n, float)
    for t in range(n):
        res[t] = _cos(I_s[t], I_m[t], eps=eps)

    # local displacement ΔI_t
    dI = np.zeros(n, float)
    for t in range(1, n):
        dI[t] = float(np.linalg.norm(I_s[t] - I_s[t - 1]))

    # robust scaling by 95th percentile
    s = float(np.quantile(dI[1:], 0.95)) if n > 2 else float(np.max(dI) + eps)
    s = max(s, 1e-6)
    dI_s = np.clip(dI / s, 0.0, 1.0)

# =========================
# Identity over trajectory 
# =========================
    identity_traj = np.zeros(n, float)
    identity_traj[0] = 1.0

    gamma = 0.7

    for t in range(1, n):
        local_identity = 0.5 * res[t] + 0.5 * (1.0 - dI_s[t])

        recovery = 0.4 * res[t] * (1.0 - identity_traj[t - 1])

        identity_traj[t] = (
            gamma * identity_traj[t - 1]
            + (1.0 - gamma) * local_identity
            + recovery
        )


    p5, p95 = np.percentile(identity_traj, [5, 95])
    identity_traj = (identity_traj - p5) / (p95 - p5 + 1e-8)
    identity_traj = np.clip(identity_traj, 0.0, 1.0)

    z = 3.0 * (identity_traj - float(b))
    C_t = _sigma(z)

    return {
        "I_s": I_s,
        "I_m": I_m,
        "res": res,
        "dI_norm": dI_s,
        "identity_traj": identity_traj,
        "C_t": np.clip(C_t, 0.0, 1.0),
    }
