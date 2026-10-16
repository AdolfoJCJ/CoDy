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

import math
from typing import Dict, Optional

import numpy as np

def compute_primary_state(
    Ct: np.ndarray,
    C_inv: Optional[np.ndarray],
    d_i: np.ndarray,
    kappa_i: np.ndarray,
    rho_t: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Construct the primary state:
        z_t = (S_t, R_t, d_t, kappa_t, u_t)

    where:
      S_t = 1 - C_t       contextual discontinuity level
      R_t = 1 - C_inv     structural instability / low persistence
      d_t                 angular displacement in embedding space
      kappa_t             turning-angle curvature of the embedding trajectory
      u_t = 1 - rho_t     local semantic dispersion

    These five coordinates are kept separate. No geometry composite is required
    for the primary analysis. If C_inv is unavailable/not yet estimable, R_t
    remains NaN and any full-state summary that requires all five coordinates is
    likewise undefined at those turns.
    """
    Ct = np.asarray(Ct, float)
    d_i = np.asarray(d_i, float)
    kappa_i = np.asarray(kappa_i, float)
    rho_t = np.asarray(rho_t, float)

    lengths = [Ct.size, d_i.size, kappa_i.size, rho_t.size]
    if C_inv is not None:
        lengths.append(np.asarray(C_inv).size)
    n = min(lengths) if lengths else 0

    Ct = Ct[:n]
    d_i = d_i[:n]
    kappa_i = kappa_i[:n]
    rho_t = rho_t[:n]

    S = np.full(n, np.nan, dtype=float)
    valid_ct = np.isfinite(Ct)
    S[valid_ct] = 1.0 - np.clip(Ct[valid_ct], 0.0, 1.0)

    R = np.full(n, np.nan, dtype=float)
    if C_inv is not None:
        C_arr = np.asarray(C_inv, float)[:n]
        valid_r = np.isfinite(C_arr)
        R[valid_r] = 1.0 - np.clip(C_arr[valid_r], 0.0, 1.0)

    d = np.full(n, np.nan, dtype=float)
    valid_d = np.isfinite(d_i)
    d[valid_d] = np.clip(d_i[valid_d], 0.0, 1.0)

    kappa = np.full(n, np.nan, dtype=float)
    valid_k = np.isfinite(kappa_i)
    kappa[valid_k] = np.clip(kappa_i[valid_k], 0.0, 1.0)

    u = np.full(n, np.nan, dtype=float)
    valid_rho = np.isfinite(rho_t)
    u[valid_rho] = 1.0 - np.clip(rho_t[valid_rho], 0.0, 1.0)

    return {
        "S_t": S,
        "R_t": R,
        "d_t": d,
        "kappa_t": kappa,
        "u_t": u,
    }

def _finite_first_difference(x: np.ndarray) -> np.ndarray:
    """First difference preserving missingness."""
    x = np.asarray(x, float)
    d = np.full(x.shape, np.nan, dtype=float)
    if x.size <= 1:
        return d
    valid = np.isfinite(x[1:]) & np.isfinite(x[:-1])
    idx = np.where(valid)[0] + 1
    d[idx] = x[idx] - x[idx - 1]
    return d

def _robust_change_scale(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Robust within-dialogue scale for a turn-to-turn change series.

    Primary estimate: 1.4826 * MAD around the median, which is comparable to
    a standard deviation under a Gaussian reference model while remaining less
    sensitive to isolated jumps. If MAD collapses to ~0, fall back to RMS about
    zero so constant non-zero drift remains calibratable. A completely flat
    dimension receives scale 1.0, making its standardized changes exactly zero.

    This scale is retrospective (it uses the full valid dialogue) and is therefore
    intended for offline descriptive comparison, not causal online detection.
    """
    x = np.asarray(x, float)
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.nan

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = 1.4826 * mad

    if (not np.isfinite(scale)) or scale <= eps:
        rms = float(np.sqrt(np.mean(vals ** 2)))
        if np.isfinite(rms) and rms > eps:
            scale = rms
        else:
            scale = 1.0

    return float(scale)

def _masked_corr(a: np.ndarray, b: np.ndarray, min_pairs: int = 4) -> float:
    """Pearson correlation using only finite paired observations."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = min(a.size, b.size)
    if n == 0:
        return np.nan
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < int(max(2, min_pairs)):
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])

def rolling_corr_series(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 7,
    min_pairs: Optional[int] = None,
) -> np.ndarray:
    """Trailing-window Pearson correlation using only observations up to t."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(x.size, y.size)
    out = np.full(n, np.nan, dtype=float)
    w = int(max(3, window))
    if min_pairs is None:
        min_pairs = max(4, int(math.ceil(0.60 * w)))
    min_pairs = int(min(min_pairs, w))
    for t in range(n):
        start = max(0, t - w + 1)
        out[t] = _masked_corr(
            x[start:t + 1], y[start:t + 1], min_pairs=min_pairs
        )
    return out

def compute_cross_dimensional_analysis(
    S: np.ndarray,
    R: np.ndarray,
    d: np.ndarray,
    kappa: np.ndarray,
    u: np.ndarray,
    rolling_window: int = 7,
) -> Dict[str, object]:
    """
    Descriptive dynamics of the primary five-dimensional state
        z_t=(S_t,R_t,d_t,kappa_t,u_t).

    No composite Geometry Driver enters this analysis. J_t, variability-adjusted
    velocity v_rel_t is a secondary summary only; primary interpretation
    should retain the vector-valued trajectories and component-specific changes.

    A second, retrospective diagnostic adjusts each component change by its own
    robust within-dialogue variability before computing movement:
        delta_rel_i(t) = delta_i(t) / robust_scale(delta_i)
        v_rel_t        = sqrt(mean_i(delta_rel_i(t)^2))
    This prevents a naturally high-variance coordinate from dominating merely
    because it has a larger dynamic range. v_rel_t is an offline descriptive
    diagnostic, not a causal detector.
    """
    dims = {
        "S": np.asarray(S, float),
        "R": np.asarray(R, float),
        "d": np.asarray(d, float),
        "kappa": np.asarray(kappa, float),
        "u": np.asarray(u, float),
    }
    n = min(arr.size for arr in dims.values()) if dims else 0
    dims = {k: v[:n] for k, v in dims.items()}
    names = list(dims.keys())

    changes = {name: _finite_first_difference(arr) for name, arr in dims.items()}

    # Five-dimensional step magnitude. This is deliberately a secondary summary:
    # equal coordinate scaling does not imply equal construct reliability.
    valid_step = np.ones(n, dtype=bool)
    for arr in changes.values():
        valid_step &= np.isfinite(arr)

    energy = np.full(n, np.nan, dtype=float)
    if n:
        stacked_sq = np.vstack([changes[name] ** 2 for name in names])
        energy[valid_step] = np.sum(stacked_sq[:, valid_step], axis=0)

    J_raw = np.full(n, np.nan, dtype=float)
    J_raw[valid_step] = np.sqrt(energy[valid_step])
    J_t = np.full(n, np.nan, dtype=float)
    J_t[valid_step] = np.clip(J_raw[valid_step] / np.sqrt(float(len(names))), 0.0, 1.0)

    # ---------------------------------------------------------
    # Variability-adjusted multivariate movement (retrospective)
    # ---------------------------------------------------------
    # Raw Euclidean movement can be dominated by a coordinate that simply has
    # greater baseline variability (notably R_t in some dialogues). To separate
    # absolute movement from relative/anomalous movement, calibrate each first
    # difference by its own robust within-dialogue change scale.
    # Calibrate every coordinate on the same set of fully observed 5D steps,
    # so early turns where R_t is unavailable cannot give the other dimensions
    # a different reference period.
    change_scales = {
        name: _robust_change_scale(changes[name][valid_step])
        for name in names
    }

    standardized_changes = {
        name: np.full(n, np.nan, dtype=float)
        for name in names
    }
    for name in names:
        scale = change_scales[name]
        if np.isfinite(scale) and scale > 0:
            mask = np.isfinite(changes[name])
            standardized_changes[name][mask] = changes[name][mask] / float(scale)

    valid_relative_step = np.ones(n, dtype=bool)
    for arr in standardized_changes.values():
        valid_relative_step &= np.isfinite(arr)

    relative_energy = np.full(n, np.nan, dtype=float)
    if n:
        stacked_rel_sq = np.vstack([standardized_changes[name] ** 2 for name in names])
        relative_energy[valid_relative_step] = np.sum(
            stacked_rel_sq[:, valid_relative_step], axis=0
        )

    # RMS standardized movement: interpretable in robust within-dialogue scale
    # units and not inflated merely by the number of dimensions.
    v_rel_t = np.full(n, np.nan, dtype=float)
    v_rel_t[valid_relative_step] = np.sqrt(
        relative_energy[valid_relative_step] / float(len(names))
    )

    relative_shares = {name: np.full(n, np.nan, dtype=float) for name in names}
    moving_rel = valid_relative_step & (relative_energy > 1e-24)
    for name in names:
        relative_shares[name][moving_rel] = (
            standardized_changes[name][moving_rel] ** 2
        ) / relative_energy[moving_rel]

    # ---------------------------------------------------------
    # Contribution reconfiguration (composition change)
    # ---------------------------------------------------------
    # Let w_t be the five-component vector of variability-adjusted movement
    # shares. CR_t is the total-variation distance between consecutive share
    # vectors:
    #       CR_t = 0.5 * sum_i |w_i(t) - w_i(t-1)|
    # It is bounded in [0, 1]. CR_t=0 means the composition of movement is
    # unchanged; values near 1 indicate an almost complete redistribution of
    # which components account for the movement. This is deliberately distinct
    # from v_rel_t: v_rel_t asks how much standardized movement occurred, while
    # CR_t asks how much the *composition* of that movement changed.
    contribution_reconfiguration = np.full(n, np.nan, dtype=float)
    if n > 1:
        share_matrix = np.vstack([relative_shares[name] for name in names]).T
        for t in range(1, n):
            prev = share_matrix[t - 1]
            curr = share_matrix[t]
            if np.all(np.isfinite(prev)) and np.all(np.isfinite(curr)):
                contribution_reconfiguration[t] = float(
                    np.clip(0.5 * np.sum(np.abs(curr - prev)), 0.0, 1.0)
                )

    directions = {name: np.full(n, np.nan, dtype=float) for name in names}
    shares = {name: np.full(n, np.nan, dtype=float) for name in names}
    moving = valid_step & (J_raw > 1e-12)
    stationary = valid_step & (J_raw <= 1e-12)
    for name in names:
        directions[name][moving] = changes[name][moving] / J_raw[moving]
        directions[name][stationary] = 0.0
        shares[name][moving] = (changes[name][moving] ** 2) / energy[moving]

    # All 10 pairwise rolling correlations for the 5D state.
    rolling_level_corr = {}
    rolling_change_corr = {}
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            key = f"{a}|{b}"
            pairs.append((a, b, key))
            rolling_level_corr[key] = rolling_corr_series(
                dims[a], dims[b], window=rolling_window
            )
            rolling_change_corr[key] = rolling_corr_series(
                changes[a], changes[b], window=rolling_window
            )

    return {
        "dimensions": dims,
        "changes": changes,
        "J_t": J_t,
        "v_rel_t": v_rel_t,
        "change_scales": change_scales,
        "standardized_changes": standardized_changes,
        "relative_shares": relative_shares,
        "contribution_reconfiguration": contribution_reconfiguration,
        "directions": directions,
        "shares": shares,
        "rolling_level_corr": rolling_level_corr,
        "rolling_change_corr": rolling_change_corr,
    }
