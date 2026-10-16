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
import plotly.graph_objects as go

def plot_ci_lines(
    turns: np.ndarray,
    ci_df: pd.DataFrame,
    height: int = 520,
    title: str = "Per-participant coherence (Ci)",
) -> go.Figure:
    turns = np.asarray(turns, dtype=int)

    fig = go.Figure()

    if ci_df is None or ci_df.shape[1] == 0:
        fig.update_layout(
            title=title,
            height=int(height),
            margin=dict(l=40, r=200, t=40, b=40),
            xaxis_title="Turn",
            yaxis_title="Ci (0–1)",
            yaxis=dict(range=[0, 1]),
        )
        return fig

    # Plot every Ci_* column
    for col in ci_df.columns:
        if not str(col).startswith("Ci_"):
            continue
        y = np.asarray(ci_df[col].to_numpy(), dtype=float)
        n = min(turns.size, y.size)
        fig.add_trace(
            go.Scatter(
                x=turns[:n],
                y=np.clip(y[:n], 0.0, 1.0),
                mode="lines",
                name=str(col).replace("Ci_", ""),
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title=title,
        height=int(height),
        margin=dict(l=40, r=200, t=40, b=40),
        xaxis_title="Turn",
        yaxis_title="Ci (0–1)",
        yaxis=dict(range=[0, 1]),
        legend=dict(
            orientation="v",
            x=1.02, xanchor="left",
            y=1.0, yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )
    return fig

def _base_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        height=int(height),
        margin=dict(l=40, r=200, t=30, b=40),
        xaxis_title="Turn",
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )
    return fig

def _hover_direction_symbol(value: float) -> str:
    """Compact sign marker for turn-level component movement."""
    if not np.isfinite(value):
        return "·"
    if value > 1e-12:
        return "↑"
    if value < -1e-12:
        return "↓"
    return "→"

def _truncate_hover_text(value: object, max_chars: int = 180) -> str:
    """Keep Plotly hover labels readable for long utterances."""
    s = "" if value is None else str(value)
    s = " ".join(s.split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"

def plot_ct_main(
    Ct: np.ndarray,
    participants: List[str],
    title: str,
    height: int,
    C_inv: Optional[np.ndarray] = None,
    turn_data: Optional[pd.DataFrame] = None,
) -> go.Figure:
    Ct = np.asarray(Ct, dtype=float)
    n = int(Ct.size)
    x = np.arange(1, n + 1, dtype=int)
    fig = _base_layout(go.Figure(), title, height)

    hover_ready = turn_data is not None and len(turn_data) == n
    customdata = None
    hovertemplate = "<b>Turn %{x}</b><br>Cₜ: %{y:.3f}<extra></extra>"

    if hover_ready:
        td = turn_data.reset_index(drop=True)
        def col(name: str, default=np.nan):
            return td[name].to_numpy() if name in td.columns else np.full(n, default, dtype=float)
        speaker = td["participant"].astype(str).to_numpy() if "participant" in td.columns else np.asarray(participants, dtype=object)
        utterance = np.asarray([_truncate_hover_text(v) for v in (td["text"].to_numpy() if "text" in td.columns else np.full(n, "", dtype=object))], dtype=object)
        v_rel = np.asarray(col("v_rel_t"), dtype=float)
        cr = np.asarray(col("CR_t"), dtype=float)
        shares = [np.asarray(col(c), dtype=float) for c in ["Vrel_share_S","Vrel_share_R","Vrel_share_d","Vrel_share_kappa","Vrel_share_u"]]
        deltas = [np.asarray(col(c), dtype=float) for c in ["dS_t","dR_t","dd_t","dkappa_t","du_t"]]
        arrows = [np.asarray([_hover_direction_symbol(v) for v in arr], dtype=object) for arr in deltas]
        customdata = np.column_stack([speaker, utterance, v_rel, cr, shares[0], arrows[0], shares[1], arrows[1], shares[2], arrows[2], shares[3], arrows[3], shares[4], arrows[4]])
        hovertemplate = (
            "<b>Turn %{x}</b> · %{customdata[0]}<br>%{customdata[1]}<br><br>"
            "<b>Continuity</b>: %{y:.3f}<br><b>Relative movement</b>: %{customdata[2]:.3f}<br>"
            "<b>Contribution reconfiguration</b>: %{customdata[3]:.3f}<br><br>"
            "<b>Relative contribution to movement</b><br>"
            "S contextual: %{customdata[4]:.1%} %{customdata[5]}<br>"
            "R structural: %{customdata[6]:.1%} %{customdata[7]}<br>"
            "d displacement: %{customdata[8]:.1%} %{customdata[9]}<br>"
            "κ curvature: %{customdata[10]:.1%} %{customdata[11]}<br>"
            "u dispersion: %{customdata[12]:.1%} %{customdata[13]}<extra></extra>"
        )

    fig.add_trace(go.Scatter(x=x, y=np.clip(Ct,0.0,1.0), mode="lines+markers", name="Cₜ (contextual continuity)", line=dict(width=3), marker=dict(size=5), customdata=customdata, hovertemplate=hovertemplate))
    if C_inv is not None:
        C_inv = np.asarray(C_inv, dtype=float)
        if C_inv.size == n:
            fig.add_trace(go.Scatter(x=x, y=np.clip(C_inv,0.0,1.0), mode="lines", name="C_inv (structural continuity)", line=dict(width=2,dash="dot"), hovertemplate="<b>Turn %{x}</b><br>C_inv: %{y:.3f}<extra></extra>"))
    for name in list(dict.fromkeys([str(p) for p in participants])):
        y=np.full(n,np.nan,float); idx=[i for i,p in enumerate(participants) if str(p)==name]
        if idx: y[idx]=Ct[idx]
        fig.add_trace(go.Scatter(x=x,y=y,mode="markers",name=f"Participant: {name}",marker=dict(size=7),hoverinfo="skip"))
    fig.update_layout(hovermode="closest")
    return _base_layout(fig,title,height)
