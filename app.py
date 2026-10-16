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

import streamlit as st
st.set_page_config(page_title="CoDy — Conversational Dynamics Framework", layout="wide")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cody.config import LABELS, GEOM_COMPACTNESS_WINDOW
from cody.embeddings import embed_texts
from cody.coherence import compute_C_inv_series, compute_ic2_dynamics, smooth_coherence
from cody.geometry import compute_ic3_geometry, compute_all_signals
from cody.multivariate import compute_primary_state, compute_cross_dimensional_analysis
from cody.participants import compute_ci_series
from cody.data import _clean_df, load_demo, dataframe_to_csv_bytes
from cody.plots import plot_ci_lines, plot_ct_main, _truncate_hover_text
from cody.reports import build_pdf_report_bytes



# =========================================================
# CACHED / UTILITY FUNCTIONS (cache-safe)
# =========================================================










# -------------------------------
# Numeric helpers
# -------------------------------



    
# -------------------------------
# Invariant-based coherence (C_inv)
# -------------------------------









# -------------------------------
# IC-II operators & helpers
# -------------------------------




# =========================================================
# Primary continuous state + cross-dimensional analysis
# =========================================================























# -------------------------------
# Data input
# -------------------------------



# ============================

# Ci plot and State plot are both available (defined elsewhere)
# plot_ct_main no longer references undefined friction/geom_* vars
# removed dead/unreachable code under _base_layout
# plot_ct_main now returns fig and includes markers/thresholds/annotations
# ============================





    




    
    





    
    



    


# -------------------------------
# Ci trajectories per participant (embedding-based)
# -------------------------------


# -------------------------------
# Plot helpers
# -------------------------------







    
# =========================================================
# PDF REPORT (Matplotlib + PdfPages) 
# =========================================================





















# ============================
# app.py — PART 3/3
# (demo + Streamlit app main + downloads)
# State_alpha slider exists and matches the call
# Undefined vars deleted (Ct_action/Ft/PDF/geom_* vars)
# ============================

# =========================
# Streamlit app
# =========================
lang = st.sidebar.selectbox(LABELS["en"]["lang"], options=["en", "es"], index=0)
L = LABELS[lang]

st.title(L["app_title"])
st.caption(L["app_subtitle"])

with st.expander(L["what_does"], expanded=False):
    st.markdown("""
This application represents conversation as a continuously evolving organizational system rather than as a static sequence of independent utterances.

Instead of analysing dialogue solely through turn-to-turn semantic similarity, CoDy estimates an evolving conversational state that changes as each new contribution reshapes the accumulated interaction.

To represent this evolving organizational state, the framework combines multiple complementary computational observables:

- **Contextual Continuity (Cₜ)** estimates how naturally each conversational turn integrates into the evolving conversational context.

- **Organizational Persistence (C_inv)** estimates the extent to which the underlying conversational organization remains stable as the dialogue develops.

- **Conversational Trajectory Dynamics** characterizes how the overall interaction evolves through semantic space, capturing changes in conversational direction that cannot be explained by local continuity alone.

CoDy keeps these observables separate rather than collapsing them into a single categorical state.

The core representation is multivariate and continuous:

z_t = (S_t, R_t, d_t, κ_t, u_t)

with **S_t = 1 - C_t** (Contextual Discontinuity), **R_t = 1 - C_inv(t)** (Structural Instability / low persistence), **d_t** as angular displacement, **κ_t** as turning-angle curvature, and **u_t = 1 - ρ_t** as local semantic dispersion. These coordinates are kept separate in the primary analysis. No composite Geometry Driver is used in the current app; the five coordinates remain separate in the primary analysis.

The framework therefore focuses on the temporal relationships among continuous conversational observables rather than imposing regime labels.


Because raw Euclidean movement can be dominated by a coordinate that simply varies more, the app also reports **v_rel(t)**, a retrospective variability-adjusted velocity. Each component change is divided by its robust within-dialogue change scale (1.4826 × MAD, with a safe RMS fallback), and the adjusted changes are combined as an RMS multivariate movement. This diagnostic asks whether a component is changing unusually strongly relative to its own baseline variability and should not be used as an online causal detector.

Unlike traditional approaches that focus primarily on neighbouring-turn similarity, CoDy represents conversation as a dynamic organizational process. Different conversational phenomena may therefore produce distinct organizational signatures even when local semantic similarity appears relatively stable.

Internally, the framework analyses the evolving trajectory through semantic space using separate geometric observables: local angular displacement, turning-angle curvature, and local semantic dispersion. The geometric observables remain separate so that distinct behaviours are not collapsed prematurely.

The goal of CoDy is not simply to measure conversational coherence, but to provide a computational representation of the temporal organization of conversation. By making this evolving organization observable, the framework enables conversational dynamics to be visualized, interpreted, and empirically investigated.
""")

    st.write("Expected columns: `turn`, `timestamp` (optional), `participant`, `text`.")
    
# =========================
# Mode: Canonical / Explore
# =========================
mode = st.sidebar.radio(
    "Mode",
    ["Canonical", "Explore"],
    index=0,
    help="Canonical = defaults fixed (clean UI). Explore = full controls."
)
IS_CANON = (mode == "Canonical")

def ui_slider(label, min_value, max_value, value, step, key=None, help=None):
    if IS_CANON:
        return value
    return st.slider(label, min_value, max_value, value, step, key=key, help=help)

def ui_checkbox(label, value=False, key=None, help=None):
    if IS_CANON:
        return value
    return st.checkbox(label, value=value, key=key, help=help)

def ui_selectbox(label, options, index=0, key=None, help=None):
    if IS_CANON:
        return options[index]
    return st.selectbox(label, options, index=index, key=key, help=help)

def ui_text_input(label, value="", key=None, help=None):
    if IS_CANON:
        return value
    return st.text_input(label, value=value, key=key, help=help)
    
with st.expander("Parameter guide", expanded=False):
    st.markdown("""
This guide follows the current sidebar structure: **Data**, **Core**, **Structure**, and **Visual**.

---

## Data

**Load demo**  
Uses the built-in example dialogue. Disable it to upload your own CSV or Excel file.

**Upload dataset**  
Expected columns: `turn`, `timestamp` (optional), `participant`, and `text`.

---

## Core

**Embeddings mode**  
Selects how turns are represented semantically.  
Options are MiniLM/SBERT (canonical) or an explicit TF-IDF baseline.

**SBERT model**  
Allows a custom Hugging Face encoder model name compatible with mean pooling. In Canonical mode, the default MiniLM model is used automatically.

**Smoothing method**  
Controls how the coherence trajectory is smoothed.  
EMA reacts faster to local changes. EWMA produces a smoother trajectory over a longer span.

**EMA alpha**  
Controls responsiveness of EMA smoothing.  
Higher alpha = more reactive. Lower alpha = smoother and more stable.

**EWMA span**  
Controls the smoothing window for EWMA.  
Higher span = smoother curve. Lower span = more local sensitivity.

---

## Structural channel

**Enable C_inv**  
Activates the structural coherence channel based on graph invariants.

**C_inv window W**  
Controls the rolling window used to build local similarity graphs.  
Larger windows produce more stable structural estimates. Smaller windows are more sensitive.

**C_inv k-NN**  
Controls how many nearest neighbors are used when building the graph.  
Higher k creates denser graphs. Lower k creates simpler, more conservative graphs.

**C_inv edge threshold**  
Filters weak graph edges.  
Higher threshold removes weaker connections and may reduce noise.

**C_inv eigenfeatures (k)**  
Controls how many spectral graph features are retained.  
Higher values capture more structural detail. Lower values emphasize global stability.

---

## Visual

**Ci method**  
Controls how participant-level coherence trajectories are computed.

**Ci alpha**  
Controls participant-level inertia.  
Higher alpha = participant trajectories preserve more past context. Lower alpha = more responsiveness.

**State alpha**  
Controls smoothing/inertia in continuous participant state trajectories.

**Public View span**  
Controls the EWMA smoothing of the simplified public S–B–R visualization.

**Show Φ thresholds in public plot**  
Displays Φ_low and Φ_high in the public-facing plot.

**Show smoothed coherence plot**  
Displays an additional smoothed coherence view.

**Overlay smoothed curve on main plot**  
Adds the smoothed Cₜ curve to the main plot.

**Overlay C_inv on main plot**  
Adds the structural coherence channel to the main plot.

---

## Practical tip

Use **Canonical mode** for clean demos and stable outputs.  
Use **Explore mode** when inspecting or tuning the available analysis parameters.
""")

with st.sidebar:
    st.header(L["params"])

    data_tab, core_tab, event_tab, visual_tab = st.tabs([
        "Data",
        "Core",
        "Structure",
        "Visual",
    ])

    with data_tab:
        use_demo = st.checkbox(L["load_demo"], value=True)
        uploaded = (
            None
            if use_demo
            else st.file_uploader(
                L["upload"],
                type=["csv", "xlsx"],
                key="single_dialogue",
            )
        )
        

    with core_tab:
        st.subheader(L["sem_repr"])

        emb_mode = ui_selectbox(
            L["emb_mode"],
            ["auto", "sbert", "tfidf"],
            index=0
        )
                

        st.markdown("### Smoothed coherence")
        smooth_method = ui_selectbox("Smoothing method", ["ema", "ewma"], index=0)
        env_alpha = ui_slider("EMA alpha", 0.05, 0.60, 0.20, 0.01)
        env_span = ui_slider("EWMA span", 3, 25, 9, 1)

        # Canonical MiniLM model; TF-IDF does not require a transformer model.
        default_model = (
            ""
            if emb_mode == "tfidf"
            else "sentence-transformers/all-MiniLM-L6-v2"
        )

        sbert_model = ui_text_input(L["sbert_model"], value=default_model)

    with event_tab:
        st.subheader("Structural channel")
        

        st.markdown("### Invariant coherence (C_inv)")
        use_cinv = ui_checkbox("Enable C_inv (graph invariants)", value=True)
        cinv_window = ui_slider("C_inv window W", 6, 24, 8, 1)
        cinv_knn = ui_slider("C_inv k-NN", 2, 10, 3, 1)
        cinv_thr = ui_slider("C_inv edge threshold", 0.00, 0.30, 0.16, 0.01)
        cinv_keigs = ui_slider("C_inv eigenfeatures (k)", 3, 12, 6, 1)

    with visual_tab:
        st.subheader("Research Tutorial view")
        ci_method = ui_selectbox(L["ci_method"], ["ctx", "im"], index=0)
        ci_alpha = ui_slider(L["ci_alpha"], 0.70, 0.99, 0.90, 0.01)

        overlay_cinv_on_main = ui_checkbox("Overlay C_inv on main plot", value=True)

        st.markdown("### Continuous multivariate dynamics")
        show_crossdim = True
        crossdim_window = ui_slider(
            "Rolling dependency window",
            4, 15, 7, 1,
        )
        st.caption(
            "Primary tutorial analysis: z_t=(S_t,R_t,d_t,κ_t,u_t), where S_t=1-C_t is Contextual "
            "Discontinuity, R_t=1-C_inv is Structural Instability / low persistence, d_t is angular "
            "displacement, κ_t is turning-angle curvature, and u_t=1-ρ_t is local semantic dispersion. "
            "The five components remain separate; no legacy geometry composite or event classifier is used."
        )

            

if use_demo:
    df = load_demo(n_turns=34)
else:
    if uploaded is None:
        st.info("Upload a dataset or enable demo.")
        st.stop()
        
    if str(uploaded.name).lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

df, missing = _clean_df(df)

# =========================
# Core dialogue arrays + embeddings
# =========================
turns = df["turn"].to_numpy(dtype=int)
texts = df["text"].astype(str).tolist()
participants = df["participant"].astype(str).tolist()

try:
    E, used_mode, emb_msg = embed_texts(
        texts=texts,
        mode=emb_mode,
        sbert_model=sbert_model,
    )
except Exception as e:
    st.error(
        "Embedding backend failed. CoDy stopped instead of silently switching "
        "to a different semantic representation."
    )
    st.code(str(e))
    st.stop()

st.caption(emb_msg)

# -------------------------------
# C_inv (graph-invariant coherence)
# -------------------------------
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

# --- Smooth C_inv for visualization only ---
C_inv_plot = None
if C_inv is not None:
    C_inv_plot = np.asarray(C_inv, float).copy()
    mask = np.isfinite(C_inv_plot)
    if mask.sum() >= 5:
        valid_idx = np.where(mask)[0]
        a, b = valid_idx[0], valid_idx[-1]
        seg = C_inv_plot[a:b+1]                       # only valid segment
        seg_s = smooth_coherence(seg, method="ema", ema_alpha=0.18, ewma_span=9)
        C_inv_plot[a:b+1] = seg_s

ic2 = compute_ic2_dynamics(
    E,
    alpha_context=0.84,
    b=0.40,
)

# 1) Raw IC-II coherence (sigmoid output)
Ct_raw = np.asarray(ic2["C_t"], float)
Ct_raw = np.clip(Ct_raw, 0.0, 1.0)

# 2) Canonical coherence used by the app everywhere
Ct_base = np.clip(Ct_raw, 0.0, 1.0)

# 3) Alignment with I_m (kept as auxiliary signal)
Ct_im = 0.5 * (1.0 + np.asarray(ic2["res"], float))
Ct_im = np.clip(Ct_im, 0.0, 1.0)

# 4) Smoothed coherence for visualization/support only
Ct_smooth = smooth_coherence(
    Ct_base,
    method=smooth_method,
    ema_alpha=float(env_alpha),
    ewma_span=int(env_span),
)
Ct_smooth = np.clip(Ct_smooth, 0.0, 1.0)

# ============================
# IC-III => IC-II (geometry)
# ============================
ic3 = compute_ic3_geometry(E=E)

signals = compute_all_signals(
    E=E,
    texts=texts,
    Ct_base=Ct_base,
    C_inv=C_inv,
    ic3=ic3
)

rho_t = signals["rho_t"]
u_t = signals["u_t"]
d_i = signals["d_i"]
kappa_i = signals["kappa_i"]


# =========================
# BUILD OUTPUT DATAFRAME
# =========================

df_out = df.copy()

# --- Core signals ---
df_out["Ct"] = Ct_base
df_out["Ct_im"] = Ct_im
df_out["rho_t"] = rho_t
df_out["d_i_raw"] = np.asarray(signals["d_i"], float)
df_out["kappa_i_raw"] = np.asarray(signals["kappa_i"], float)
df_out["d_t"] = np.asarray(d_i, float)
df_out["kappa_t"] = np.asarray(kappa_i, float)
df_out["u_t"] = np.asarray(u_t, float)

# =========================
# C_inv (if available)
# =========================
if C_inv is not None and np.asarray(C_inv).size == len(df_out):
    df_out["C_inv"] = np.asarray(C_inv, float)
else:
    df_out["C_inv"] = np.nan

# =========================================================
# PRIMARY CONTINUOUS COMPUTATIONAL STATE
# =========================================================
# Primary Research Tutorial state: z_t=(S_t,R_t,d_t,kappa_t,u_t).
primary_state = compute_primary_state(
    Ct=np.asarray(Ct_base, float),
    C_inv=(np.asarray(C_inv, float) if C_inv is not None else None),
    d_i=np.asarray(d_i, float),
    kappa_i=np.asarray(kappa_i, float),
    rho_t=np.asarray(rho_t, float),
)

df_out["S_t"] = primary_state["S_t"]
df_out["R_t"] = primary_state["R_t"]
df_out["d_t"] = primary_state["d_t"]
df_out["kappa_t"] = primary_state["kappa_t"]
df_out["u_t"] = primary_state["u_t"]

# =========================================================
# PRIMARY CROSS-DIMENSIONAL ANALYSIS
# =========================================================
crossdim = compute_cross_dimensional_analysis(
    S=df_out["S_t"].to_numpy(dtype=float),
    R=df_out["R_t"].to_numpy(dtype=float),
    d=df_out["d_t"].to_numpy(dtype=float),
    kappa=df_out["kappa_t"].to_numpy(dtype=float),
    u=df_out["u_t"].to_numpy(dtype=float),
    rolling_window=int(crossdim_window),
)

# First differences
change_col_map = {
    "S": "dS_t",
    "R": "dR_t",
    "d": "dd_t",
    "kappa": "dkappa_t",
    "u": "du_t",
}
for name, col in change_col_map.items():
    df_out[col] = crossdim["changes"][name]

# Secondary movement summaries
df_out["J_t"] = crossdim["J_t"]
df_out["v_rel_t"] = crossdim["v_rel_t"]
df_out["CR_t"] = crossdim["contribution_reconfiguration"]
for name, col in {
    "S": "J_share_S",
    "R": "J_share_R",
    "d": "J_share_d",
    "kappa": "J_share_kappa",
    "u": "J_share_u",
}.items():
    df_out[col] = crossdim["shares"][name]

for name, col in {
    "S": "Vrel_share_S",
    "R": "Vrel_share_R",
    "d": "Vrel_share_d",
    "kappa": "Vrel_share_kappa",
    "u": "Vrel_share_u",
}.items():
    df_out[col] = crossdim["relative_shares"][name]

for name, col in {
    "S": "dir_S",
    "R": "dir_R",
    "d": "dir_d",
    "kappa": "dir_kappa",
    "u": "dir_u",
}.items():
    df_out[col] = crossdim["directions"][name]

# Export all pairwise local coupling series under stable column names.
pair_alias = {"S": "S", "R": "R", "d": "d", "kappa": "kappa", "u": "u"}
for key, arr in crossdim["rolling_change_corr"].items():
    a, b = key.split("|")
    df_out[f"r_d{pair_alias[a]}_d{pair_alias[b]}"] = arr

fig_main = plot_ct_main(
    Ct=Ct_base,
    participants=participants,
    title="CoDy - Continuity Dynamics",
    height=560,
    C_inv=(
        C_inv
        if overlay_cinv_on_main and C_inv is not None
        else None
    ),
    turn_data=df_out,
)

st.plotly_chart(fig_main, use_container_width=True)
st.caption(
    "Hover any turn on Cₜ to inspect the utterance, variability-adjusted movement v_rel(t), contribution "
    "reconfiguration CR_t, and the relative contribution of S, R, d, κ and u. Arrows indicate whether each "
    "component increased, decreased or remained approximately unchanged at that turn. CR_t measures redistribution "
    "of the component-share vector from the previous valid turn (0 = same composition; 1 = maximal redistribution), "
    "so it should be interpreted separately from movement magnitude."
)



ci_df = compute_ci_series(
    E=E,
    participants=participants,
    method=ci_method,
    alpha=float(ci_alpha),
)

if ci_df is not None and ci_df.shape[1] > 0:
    for col in ci_df.columns:
        df_out[col] = ci_df[col].to_numpy()

# =========================
# FULL DIALOGUE TABLE
# =========================
st.subheader(L["table_title"])

ci_cols = [c for c in df_out.columns if str(c).startswith("Ci_")]
cols_order = [
    "turn", "participant", "text",
    "Ct", "Ct_im", "C_inv", *ci_cols,
    "S_t", "R_t", "d_t", "kappa_t", "u_t",
    "dS_t", "dR_t", "dd_t", "dkappa_t", "du_t",
    "J_t", "v_rel_t", "CR_t",
    "J_share_S", "J_share_R", "J_share_d", "J_share_kappa", "J_share_u",
    "Vrel_share_S", "Vrel_share_R", "Vrel_share_d", "Vrel_share_kappa", "Vrel_share_u",
    "rho_t",
]
cols_show = [c for c in cols_order if c in df_out.columns]
st.dataframe(df_out[cols_show], use_container_width=True)
st.caption(
    "Primary turn-level state: z_t=(S_t,R_t,d_t,κ_t,u_t). S_t=1-C_t is Contextual Discontinuity; "
    "R_t=1-C_inv is Structural Instability / low persistence; d_t is angular displacement; κ_t is "
    "turning-angle curvature; and u_t=1-ρ_t is local semantic dispersion. v_rel_t is a retrospective "
    "robust-scale-adjusted movement diagnostic that reduces domination "
    "by intrinsically high-variance coordinates. CR_t is the total-variation distance between consecutive vectors of "
    "variability-adjusted component shares and therefore measures compositional redistribution rather than movement "
    "magnitude."
)

# =========================================================
# Research Tutorial — Primary Multicomponent Dynamics
# =========================================================
if show_crossdim:
    with st.expander("Research Tutorial — Primary Multicomponent Dynamics", expanded=True):
        st.caption(
            "Primary analysis keeps five continuous observables separate: z_t=(S_t,R_t,d_t,κ_t,u_t). "
            "This avoids repeating the Transition Pressure mistake of assuming that distinct signals should be "
            "collapsed into one scalar before their relationships are empirically established. d_t, κ_t and u_t "
            "are temporally causal; S_t and R_t remain offline within-dialogue descriptors because C_t and C_inv "
            "use dialogue-level scaling. Human consensus should remain an external validation target."
        )

        primary_series = {
            "S": np.asarray(df_out["S_t"], float),
            "R": np.asarray(df_out["R_t"], float),
            "d": np.asarray(df_out["d_t"], float),
            "κ": np.asarray(df_out["kappa_t"], float),
            "u": np.asarray(df_out["u_t"], float),
        }
        finite_R = int(np.isfinite(primary_series["R"]).sum())
        if finite_R < 4:
            st.warning(
                "R_t has too few valid observations for a full five-dimensional analysis. "
                "Enable C_inv and use a dialogue long enough to fill the structural window."
            )

        tab_dyn, tab_dep = st.tabs([
            "State & Change", "Dynamic Coupling"
        ])

        with tab_dyn:
            x_cd = np.arange(1, len(df_out) + 1)

            fig_state = go.Figure()
            state_labels = {
                "S": "S_t — Contextual Discontinuity (1-C_t)",
                "R": "R_t — Structural Instability (1-C_inv)",
                "d": "d_t — Angular displacement",
                "κ": "κ_t — Turning-angle curvature",
                "u": "u_t — Local semantic dispersion (1-ρ_t)",
            }
            for key, y in primary_series.items():
                fig_state.add_trace(go.Scatter(x=x_cd, y=y, mode="lines", name=state_labels[key]))
            fig_state.update_layout(
                title="Primary five-dimensional computational state",
                xaxis_title="Turn", yaxis_title="State value (0–1)",
                yaxis=dict(range=[0, 1]), height=470,
                margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_state, use_container_width=True)
            st.caption(
                "The five curves are primary observables, not components of a pre-weighted transition score. "
                "S and R describe contextual/structural state levels; d, κ and u describe distinct aspects of the "
                "embedding trajectory. Their value lies partly in possible dissociations: a human transition may be "
                "associated with only one or a particular combination rather than with all dimensions rising together."
            )

            fig_delta = go.Figure()
            delta_specs = [
                ("dS_t", "ΔS_t"), ("dR_t", "ΔR_t"), ("dd_t", "Δd_t"),
                ("dkappa_t", "Δκ_t"), ("du_t", "Δu_t"),
            ]
            for col, label in delta_specs:
                fig_delta.add_trace(go.Scatter(
                    x=x_cd, y=np.asarray(df_out[col], float), mode="lines", name=label
                ))
            fig_delta.update_layout(
                title="Turn-to-turn changes in the five primary dimensions",
                xaxis_title="Turn", yaxis_title="Change", height=470,
                margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_delta, use_container_width=True)
            st.caption(
                "First differences show directional movement rather than absolute level. These component-wise curves "
                "are more important than any composite magnitude for testing whether human-perceived transitions have "
                "a recurrent multivariate temporal signature."
            )

            fig_v_rel = go.Figure()
            fig_v_rel.add_trace(go.Scatter(
                x=x_cd, y=np.asarray(df_out["v_rel_t"], float), mode="lines",
                name="v_rel(t) — variability-adjusted movement"
            ))
            fig_v_rel.update_layout(
                title="Variability-adjusted multivariate velocity",
                xaxis_title="Turn", yaxis_title="v_rel(t) — robust-scale RMS",
                yaxis=dict(rangemode="tozero"), height=360,
                margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_v_rel, use_container_width=True)
            st.caption(
                "v_rel(t) rescales each component change by that component's robust within-dialogue variability "
                "before combining the five standardized changes as an RMS magnitude. This asks whether movement is "
                "large relative to each coordinate's own baseline dynamics, so a naturally volatile R_t cannot dominate "
                "solely because its raw range is larger. The calibration uses the full dialogue and is therefore "
                "retrospective/offline; it is a diagnostic for analysis, not an online transition detector."
            )

            scale_rows = []
            pretty_scale_names = {"S": "ΔS", "R": "ΔR", "d": "Δd", "kappa": "Δκ", "u": "Δu"}
            for key in ["S", "R", "d", "kappa", "u"]:
                scale_rows.append({
                    "Component change": pretty_scale_names[key],
                    "Robust scale": float(crossdim["change_scales"][key])
                    if np.isfinite(crossdim["change_scales"][key]) else np.nan,
                })
            with st.container():
                st.markdown("#### Variability calibration used by v_rel(t)")
                st.dataframe(pd.DataFrame(scale_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "Primary scale = 1.4826 × MAD of the component's valid first differences. If MAD is effectively "
                    "zero, the implementation falls back to RMS about zero; a completely flat component receives "
                    "scale 1 so its standardized changes remain exactly zero."
                )

            fig_rel_comp = go.Figure()
            for col, label in [
                ("Vrel_share_S", "S relative contribution"), ("Vrel_share_R", "R relative contribution"),
                ("Vrel_share_d", "d relative contribution"), ("Vrel_share_kappa", "κ relative contribution"),
                ("Vrel_share_u", "u relative contribution"),
            ]:
                fig_rel_comp.add_trace(go.Scatter(
                    x=x_cd, y=np.asarray(df_out[col], float), mode="lines",
                    stackgroup="relative_movement", name=label
                ))
            fig_rel_comp.update_layout(
                title="Composition of variability-adjusted movement",
                xaxis_title="Turn", yaxis_title="Share of standardized squared change",
                yaxis=dict(range=[0, 1]), height=390,
                margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_rel_comp, use_container_width=True)
            st.caption(
                "Relative shares are computed after dividing each component change by its own robust variability scale. "
                "The stacked profile therefore shows which component accounts for the standardized movement at each turn. "
                "This is the preferred composition view for the Research Tutorial's multivariate temporal analysis."
            )

            fig_cr = go.Figure()
            fig_cr.add_trace(go.Scatter(
                x=x_cd,
                y=np.asarray(df_out["CR_t"], float),
                mode="lines+markers",
                name="CRₜ — contribution reconfiguration",
                line=dict(width=2.5),
                marker=dict(size=5),
                customdata=np.column_stack([
                    df_out["participant"].astype(str).to_numpy(),
                    np.asarray([_truncate_hover_text(v, 140) for v in df_out["text"].to_numpy()], dtype=object),
                    np.asarray(df_out["v_rel_t"], float),
                ]),
                hovertemplate=(
                    "<b>Turn %{x}</b> · %{customdata[0]}<br>"
                    "%{customdata[1]}<br><br>"
                    "CRₜ: %{y:.3f}<br>"
                    "v_rel(t): %{customdata[2]:.3f}<extra></extra>"
                ),
            ))
            fig_cr.update_layout(
                title="Contribution reconfiguration across turns",
                xaxis_title="Turn",
                yaxis_title="CRₜ (0–1)",
                yaxis=dict(range=[0, 1]),
                height=330,
                margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_cr, use_container_width=True)
            st.caption(
                "CRₜ = ½ Σᵢ|wᵢ(t)−wᵢ(t−1)|, where wᵢ are the variability-adjusted component shares. "
                "CRₜ=0 means the composition of movement is unchanged from the previous valid turn; larger values "
                "mean that the identity of the components driving change has been redistributed more strongly. "
                "Read CRₜ together with v_rel(t): one describes composition, the other magnitude."
            )

        with tab_dep:
            x_cd = np.arange(1, len(df_out) + 1)
            rolling_change_corr = crossdim["rolling_change_corr"]
            pair_pretty = {
                "S|R": "r(ΔS,ΔR)", "S|d": "r(ΔS,Δd)", "S|kappa": "r(ΔS,Δκ)",
                "S|u": "r(ΔS,Δu)", "R|d": "r(ΔR,Δd)", "R|kappa": "r(ΔR,Δκ)",
                "R|u": "r(ΔR,Δu)", "d|kappa": "r(Δd,Δκ)", "d|u": "r(Δd,Δu)",
                "kappa|u": "r(Δκ,Δu)",
            }
            pair_options = list(rolling_change_corr.keys())
            default_pairs = [k for k in ["S|R", "S|d", "S|kappa", "R|d", "R|kappa", "d|kappa"] if k in pair_options]
            selected_pairs = st.multiselect(
                "Pairs shown in coupling plot",
                options=pair_options,
                default=default_pairs,
                format_func=lambda k: pair_pretty.get(k, k),
                key="coupling_pairs_5d",
            )
            fig_dep_delta = go.Figure()
            for key in selected_pairs:
                fig_dep_delta.add_trace(go.Scatter(
                    x=x_cd, y=np.asarray(rolling_change_corr[key], float), mode="lines",
                    name=pair_pretty.get(key, key)
                ))
            fig_dep_delta.update_layout(
                title=f"Trailing coupling among changes — window={int(crossdim_window)} turns",
                xaxis_title="Turn", yaxis_title="Pearson r", yaxis=dict(range=[-1, 1]),
                height=460, margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_dep_delta, use_container_width=True)
            st.caption(
                f"Pairwise trailing correlations over {int(crossdim_window)} turns describe local coordination among "
                "first differences. Positive values indicate same-direction movement, negative values opposing movement, "
                "and values near zero little linear coupling. The plot is selectable because ten pairwise relationships "
                "exist in a five-dimensional state."
            )




st.session_state["last_main_fig"] = fig_main
html = fig_main.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
st.download_button(
    label="Download main plot (HTML)",
    data=html,
    file_name="cody_main_plot.html",
    mime="text/html",
)





ic2_df = pd.DataFrame({
    "turn": turns,
    "res": np.asarray(ic2["res"], float),
    "dI_norm": np.asarray(ic2["dI_norm"], float),
    "Ct": np.asarray(ic2["C_t"], float),
})

ic3_df = pd.DataFrame({
    "turn": turns,
    "d_t": np.asarray(d_i, float),
    "kappa_t": np.asarray(kappa_i, float),
    "rho_t": np.asarray(rho_t, float),
    "u_t": np.asarray(u_t, float),
})

# =========================
# PDF Report (English) — Download
# =========================
st.subheader("Report (PDF)")

report_params = {
    "mode": mode,
    "emb_mode": emb_mode,
    "sbert_model": sbert_model,
    "smooth_method": smooth_method,
    "env_alpha": float(env_alpha),
    "env_span": int(env_span),
    "use_cinv": bool(use_cinv),
    "cinv_window": int(cinv_window),
    "cinv_knn": int(cinv_knn),
    "cinv_thr": float(cinv_thr),
    "cinv_keigs": int(cinv_keigs),
    "ci_method": ci_method,
    "ci_alpha": float(ci_alpha),
    "crossdim_window": int(crossdim_window),
    "primary_state": "S/R/d/kappa/u",
    "geometry_composite_primary": False,
    "geom_compactness_window": int(GEOM_COMPACTNESS_WINDOW),
}

pdf_bytes = build_pdf_report_bytes(
    df_in=df,
    df_out=df_out,
    ic2_df=ic2_df,
    ic3_df=ic3_df,
    Ct_base=Ct_base,
    Ct_smooth=Ct_smooth,
    C_inv=(C_inv if (use_cinv and C_inv is not None) else None),
    used_mode=used_mode,
    emb_msg=emb_msg,
    params=report_params,
)

st.download_button(
    label="Download report (PDF)",
    data=pdf_bytes,
    file_name="cody_report.pdf",
    mime="application/pdf",
)
st.caption("This PDF includes the run summary, parameters, continuous-dynamics plots, and the largest local Cₜ decreases (English).")

st.subheader("Downloads")
cA, cC, cD = st.columns(3)

with cA:
    st.download_button(
        L["download_full_csv"],
        data=dataframe_to_csv_bytes(df_out),
        file_name="cody_full_results.csv",
        mime="text/csv"
    )

with cC:
    st.download_button(
        L["download_ic2_csv"],
        data=dataframe_to_csv_bytes(ic2_df),
        file_name="cody_ic2_dynamics.csv",
        mime="text/csv"
    )

with cD:
    st.download_button(
        L["download_ic3_csv"],
        data=dataframe_to_csv_bytes(ic3_df),
        file_name="cody_ic3_geometry.csv",
        mime="text/csv"
    )