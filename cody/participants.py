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

from typing import List

import numpy as np
import pandas as pd

from .coherence import _cos

def compute_ci_series(
    E: np.ndarray,
    participants: List[str],
    method: str = "ctx",
    alpha: float = 0.90,
    eps: float = 1e-12,
) -> pd.DataFrame:
    E = np.asarray(E, float)
    n = E.shape[0]
    parts = [str(p) for p in participants]
    uniq = list(dict.fromkeys(parts))
    out = pd.DataFrame(index=np.arange(n))

    if n == 0 or len(uniq) == 0:
        return out

    e = np.zeros_like(E, float)
    for i in range(n):
        v = E[i]
        nv = float(np.linalg.norm(v))
        e[i] = v / (nv + eps) if nv > eps else v * 0.0

    method = (method or "ctx").lower().strip()

    if method == "im":
        for p in uniq:
            idx = [i for i, pp in enumerate(parts) if pp == p]
            if not idx:
                continue
            centroid = np.mean(e[idx], axis=0)
            centroid /= (float(np.linalg.norm(centroid)) + eps)
            ci = np.zeros(n, float)
            for t in range(n):
                ci[t] = 0.5 * (1.0 + _cos(e[t], centroid, eps=eps))
            out[f"Ci_{p}"] = np.clip(ci, 0.0, 1.0)
        return out

    a = float(np.clip(alpha, 0.0, 0.999))
    ctx = {p: e[0].copy() for p in uniq}
    ci_cols = {p: np.zeros(n, float) for p in uniq}

    for t in range(n):
        speaker = parts[t]
        for p in uniq:
            ci_cols[p][t] = 0.5 * (1.0 + _cos(e[t], ctx[p], eps=eps))

        c_prev = ctx[speaker]
        u = a * c_prev + (1.0 - a) * e[t]
        nu = float(np.linalg.norm(u))
        ctx[speaker] = (u / (nu + eps)) if nu > eps else c_prev

    for p in uniq:
        out[f"Ci_{p}"] = np.clip(ci_cols[p], 0.0, 1.0)

    return out
