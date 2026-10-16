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

from typing import Dict

# ============================
# app.py - CoDy continuous multivariate analysis application
# ============================

# =====================
# Bilingual labels
# =====================
LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "📈 Conversational Dynamics Framework 📉",
        "app_subtitle": "Developed by CNøde — Research in Informational Systems",
        "what_does": "What does CoDy do?",
        "params": "Parameters",
        "load_demo": "Load demo",
        "expected_cols": "Expected columns: turn, timestamp (optional), participant, text.",
        "upload": "Upload a dataset (.csv or .xlsx)",
        "sem_repr": "Semantic representation",
        "emb_mode": "Embeddings mode",
        "sbert_model": "SBERT model (optional)",
        "coh_mode": "Coherence mode",
        "coh_form": "Coherence formulation",
        "ic2": "IC-II coherence dynamics",
        "geom": "IC–III geometric layer",
        "smooth": "Cₜ smoothing",
        "phi": "Φ thresholds (percentiles by default)",
        "layout": "Layout",
        "compute": "Compute metrics",
        "preview": "Dialogue preview",
        "global_metrics": "Global metrics",
        "table_title": "Turn-Level Dialogue Analysis",
        "overview": "Overview plot (CoDy)",
        "geom_plot": "IC–III geometric layer: dᵢ and κᵢ",
        "csv": "Download CSV results",
        "download_full_csv": "Download full results (CSV)",
        "download_ic2_csv": "Download IC–IIa dynamics (CSV)",
        "download_ic3_csv": "Download IC–III geometric layer (CSV)",
        "debug_header": "Debug",
        "lang": "Language",
        "public_header": "Public View (Smoothed S–T–B)",
        "public_span": "Smoothing span (EWMA)",
        "public_show_thresholds": "Show Φ thresholds in public plot",
        "public_title": "Public plot: Smoothed coherence + S–B–R regimes",
        "ci_header": "Participant trajectories (Ci)",
        "ci_alpha": "Ci context inertia α (per-participant)",
        "ci_method": "Ci method",
        "ci_title": "Per-participant coherence trajectories (Ci)",
        "state_alpha": "State trajectory inertia α (continuous lines)",
        "state_title": "Per-participant continuous trajectories (state) + Φ thresholds",
    },
    "es": {
        "app_title": "📈 Conversational Dynamics Framework 📉",
        "app_subtitle": "Developed by CNøde — Research in Informational Systems",
        "what_does": "¿Qué hace esta app?",
        "params": "Parámetros",
        "load_demo": "Cargar demo",
        "expected_cols": "Columnas esperadas: turn, timestamp (opcional), participant, text.",
        "upload": "Sube un dataset (.csv o .xlsx)",
        "sem_repr": "Representación semántica",
        "emb_mode": "Modo de embeddings",
        "sbert_model": "Modelo SBERT (opcional)",
        "coh_mode": "Modo de coherencia",
        "coh_form": "Formulación de coherencia",
        "ic2": "Dinámica de coherencia IC-II",
        "geom": "Capa geométrica IC–III",
        "smooth": "Suavizado de Cₜ",
        "phi": "Umbrales Φ (por defecto percentiles)",
        "layout": "Layout",
        "compute": "Calcular métricas",
        "preview": "Vista previa del diálogo",
        "global_metrics": "Métricas globales",
        "table_title": "Análisis dinámico por turno",
        "overview": "Plot overview (CoDy)",
        "geom_plot": "Capa geométrica IC–III: dᵢ y κᵢ",
        "csv": "Descargar resultados CSV",
        "download_full_csv": "Descargar resultados completos (CSV)",
        "download_ic2_csv": "Descargar dinámica IC–IIa (CSV)",
        "download_ic3_csv": "Descargar capa geométrica IC–III (CSV)",
        "ctx_header": "Coherencia context-aware (Ct_new)",
        "debug_header": "Debug",
        "lang": "Idioma",
        "public_header": "Vista pública (S–B–R suavizado)",
        "public_span": "Span de suavizado (EWMA)",
        "public_show_thresholds": "Mostrar umbrales Φ en plot público",
        "public_title": "Plot público: coherencia suavizada + regímenes S–B–R",
        "ci_header": "Trayectorias por participante (Ci)",
        "ci_alpha": "Inercia de contexto α (Ci por participante)",
        "ci_method": "Método Ci",
        "ci_title": "Trayectorias de coherencia por participante (Ci) + umbrales Φ",
        "state_alpha": "Inercia de trayectoria α (líneas continuas)",
        "state_title": "Trayectorias continuas por participante (state) + umbrales Φ",
    }
}

# =========================================================
# Geometric observable configuration
# =========================================================
GEOM_COMPACTNESS_WINDOW = 2
