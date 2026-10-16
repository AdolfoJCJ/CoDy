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

"""Reusable, Streamlit-independent CoDy core pipeline.

The public app does not need to use this wrapper internally. It exists so the
same computational path can be called from notebooks, scripts, and future
replication analyses without importing the Streamlit UI.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .embeddings import embed_texts
from .coherence import compute_C_inv_series, compute_ic2_dynamics, smooth_coherence
from .geometry import compute_ic3_geometry, compute_all_signals
from .multivariate import compute_primary_state, compute_cross_dimensional_analysis


def run_core_pipeline(
    df: pd.DataFrame,
    *,
    emb_mode: str = "sbert",
    sbert_model: Optional[str] = None,
    use_cinv: bool = True,
    cinv_window: int = 8,
    cinv_knn: int = 5,
    cinv_thr: float = 0.10,
    cinv_keigs: int = 6,
    smooth_method: str = "ema",
    ema_alpha: float = 0.18,
    ewma_span: int = 9,
    rolling_window: int = 7,
):
    """Run the current continuous CoDy computation and return arrays + dataframe."""
    texts = df["text"].astype(str).tolist()

    E, used_mode, emb_msg = embed_texts(
        texts=texts,
        mode=emb_mode,
        sbert_model=sbert_model,
    )

    C_inv = None
    if use_cinv:
        C_inv = compute_C_inv_series(
            E,
            window=int(cinv_window),
            k_nn=int(cinv_knn),
            thr=float(cinv_thr),
            k_eigs=int(cinv_keigs),
            D_max=None,
        )

    ic2 = compute_ic2_dynamics(E, alpha_context=0.84, b=0.40)
    Ct_base = np.clip(np.asarray(ic2["C_t"], float), 0.0, 1.0)
    Ct_im = np.clip(0.5 * (1.0 + np.asarray(ic2["res"], float)), 0.0, 1.0)
    Ct_smooth = np.clip(
        smooth_coherence(
            Ct_base,
            method=smooth_method,
            ema_alpha=float(ema_alpha),
            ewma_span=int(ewma_span),
        ),
        0.0,
        1.0,
    )

    ic3 = compute_ic3_geometry(E=E)
    signals = compute_all_signals(
        E=E,
        texts=texts,
        Ct_base=Ct_base,
        C_inv=C_inv,
        ic3=ic3,
    )

    primary_state = compute_primary_state(
        Ct=Ct_base,
        C_inv=(np.asarray(C_inv, float) if C_inv is not None else None),
        d_i=np.asarray(signals["d_i"], float),
        kappa_i=np.asarray(signals["kappa_i"], float),
        rho_t=np.asarray(signals["rho_t"], float),
    )

    crossdim = compute_cross_dimensional_analysis(
        S=primary_state["S_t"],
        R=primary_state["R_t"],
        d=primary_state["d_t"],
        kappa=primary_state["kappa_t"],
        u=primary_state["u_t"],
        rolling_window=int(rolling_window),
    )

    out = df.copy()
    out["Ct"] = Ct_base
    out["Ct_im"] = Ct_im
    out["rho_t"] = signals["rho_t"]
    out["d_i_raw"] = signals["d_i"]
    out["kappa_i_raw"] = signals["kappa_i"]
    for col in ("S_t", "R_t", "d_t", "kappa_t", "u_t"):
        out[col] = primary_state[col]
    out["C_inv"] = np.asarray(C_inv, float) if C_inv is not None else np.nan

    change_col_map = {
        "S": "dS_t",
        "R": "dR_t",
        "d": "dd_t",
        "kappa": "dkappa_t",
        "u": "du_t",
    }
    for name, col in change_col_map.items():
        out[col] = crossdim["changes"][name]

    out["J_t"] = crossdim["J_t"]
    out["v_rel_t"] = crossdim["v_rel_t"]
    out["CR_t"] = crossdim["contribution_reconfiguration"]

    for name in ("S", "R", "d", "kappa", "u"):
        out[f"J_share_{name}"] = crossdim["shares"][name]
        out[f"Vrel_share_{name}"] = crossdim["relative_shares"][name]
        out[f"dir_{name}"] = crossdim["directions"][name]

    return {
        "df_out": out,
        "E": E,
        "used_mode": used_mode,
        "emb_msg": emb_msg,
        "ic2": ic2,
        "ic3": ic3,
        "Ct_base": Ct_base,
        "Ct_smooth": Ct_smooth,
        "C_inv": C_inv,
        "signals": signals,
        "primary_state": primary_state,
        "crossdim": crossdim,
    }
