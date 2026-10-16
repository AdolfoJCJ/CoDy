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

from io import BytesIO
from typing import List, Tuple
import re

import numpy as np
import pandas as pd

REQUIRED_COLS = ["turn", "participant", "text"]

def _clean_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    for c in missing:
        if c == "turn":
            df[c] = np.arange(1, len(df) + 1, dtype=int)
        else:
            df[c] = ""
    if "timestamp" not in df.columns:
        df["timestamp"] = ""
    try:
        df["turn"] = df["turn"].astype(int)
    except Exception:
        df["turn"] = pd.to_numeric(df["turn"], errors="coerce").fillna(0).astype(int)
    df["participant"] = df["participant"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["turn"] > 0].sort_values("turn").reset_index(drop=True)
    return df, missing

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to downloadable UTF-8 CSV bytes once."""
    if df is None or df.empty:
        return b""
    buffer = BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    buffer.seek(0)
    return buffer.getvalue()

def read_dialogue_file(uploaded_file) -> pd.DataFrame:
    """Read and clean one uploaded CSV or XLSX dialogue."""
    name = str(uploaded_file.name).lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

    df, missing = _clean_df(df)

    if df.empty:
        raise ValueError("The file contains no valid dialogue turns.")

    return df

def infer_dialog_id(uploaded_file, df: pd.DataFrame) -> str:
    """
    Prefer a dialog_id column. Otherwise use the filename.
    """
    if "dialog_id" in df.columns:
        values = df["dialog_id"].dropna().astype(str).str.strip()
        values = values[values != ""]

        if not values.empty:
            return values.iloc[0]

    filename = str(uploaded_file.name)
    return re.sub(r"\.(csv|xlsx)$", "", filename, flags=re.IGNORECASE)

def load_demo(n_turns: int = 36) -> pd.DataFrame:
    speakers = ["A", "B", "C", "D"]
    turns, parts, texts = [], [], []

    all_lines = [
        # --- Stable opening: planning the explanation ---
        "Before presenting the app, we should explain that it tracks how conversational coherence changes over time.",
        "Yes, and we should make clear that it models dialogue as a dynamic process rather than a static exchange of sentences.",
        "Exactly. The point is that each turn either sustains or destabilizes the evolving trajectory of the conversation.",
        "So the app is not just reading text. It is estimating how each contribution fits into a shared conversational structure.",
        "One useful angle is to say that semantic coherence measures how strongly each turn aligns with the evolving context.",
        "And structural coherence adds another layer by capturing changes in the relational organization of the dialogue.",
        "Right, so the system can distinguish ordinary continuation from deeper forms of disruption.",
        "That distinction is probably what makes the tool most interesting for analysis.",

        # --- Stable elaboration ---
        "We should also mention that the output is visual, so people can see where coherence rises, drops, and stabilizes again.",
        "Yes, because the value is not just a final score but the temporal pattern across turns.",
        "That makes the app useful for inspecting where the continuous dynamics reorganize and for comparing those regions with external human judgments.",
        "Exactly. It gives a structured view of dialogue as something measurable in time.",

        # --- Local topic shift ---
        "Did anyone remember to buy coffee filters for tomorrow morning?",
        "That feels unrelated. We were still defining what the app actually measures.",
        "Yes, that move shifts the topic away from the explanation we were building.",
        "Let us return to the app itself and leave practical things for later.",

        # --- Return to the prior explanatory trajectory ---
        "Right. Back to the app: another key point is that coherence is tracked turn by turn.",
        "And that allows us to compare more stable stretches with stretches showing larger changes in the continuous observables.",
        "We should probably say that the system supports interpretation rather than replacing human judgment.",
        "Yes, because the graphs are useful precisely when they can later be compared with annotation or close reading.",

        # --- Stable continuation ---
        "We can also mention that the app compares participants, not only the dialogue as a whole.",
        "That matters because participant trajectories can diverge even when the dialogue-level trajectory looks relatively stable.",
        "So the contribution is both global and participant-sensitive.",
        "Exactly. That makes the analysis richer than using one overall average.",

        # --- Strong frame shift ---
        "Unless coherence is actually controlled by a hidden committee of underwater mathematicians rewriting the dialogue in secret.",
        "Okay, that breaks the frame much more strongly.",
        "Yes, that is no longer a simple topic drift. It disrupts the explanatory frame itself.",
        "Let us reset and return to the task in a grounded way.",

        # --- Return to the prior frame ---
        "Fine. The central claim is that the app represents conversational organization continuously across multiple complementary signals.",
        "And it does so by combining semantic continuity with structural variation over time.",
        "Keeping the signals separate avoids forcing different forms of change into a single event category.",
        "We should end by saying that this makes dialogue dynamics observable rather than merely intuitive.",

        # --- Stable closing ---
        "So the simplest explanation is that the app represents how conversational organization changes over time across several continuous dimensions.",
        "And because it does that turn by turn, it becomes possible to inspect the dynamics rather than only the outcome.",
        "That is probably the clearest closing: the app makes conversational structure visible.",
        "Agreed. That gives us a concise and realistic summary of what the system does.",
    ]

    all_lines = all_lines[:max(1, int(n_turns))]

    for i, text in enumerate(all_lines):
        turns.append(i + 1)
        parts.append(speakers[i % len(speakers)])
        texts.append(text)

    return pd.DataFrame(
        {
            "turn": np.array(turns, dtype=int),
            "timestamp": "",
            "participant": parts,
            "text": texts,
        }
    )        
