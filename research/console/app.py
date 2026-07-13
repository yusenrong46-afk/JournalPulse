from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "research" / "experiments"

st.set_page_config(page_title="JournalPulse Research Console", page_icon="JP", layout="wide")
st.title("JournalPulse research console")
st.caption("Internal evidence workspace. Empty panels mean evidence has not been produced yet.")


def load_records(kind: str) -> list[dict]:
    path = RUNS / kind
    if not path.exists():
        return []
    records: list[dict] = []
    for artifact in sorted(path.glob("*.json")):
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            payload["artifact"] = artifact.name
            records.append(payload)
        except (json.JSONDecodeError, OSError):
            continue
    return records


def evidence_table(kind: str, empty_message: str) -> list[dict]:
    records = load_records(kind)
    if records:
        st.dataframe(pd.json_normalize(records), width="stretch", hide_index=True)
    else:
        st.info(empty_message)
    return records


page = st.sidebar.radio(
    "Workspace",
    [
        "Release gates",
        "Simulator",
        "Policy comparison",
        "LLM evaluation",
        "Memory retrieval",
        "Safety suite",
        "Experiment registry",
    ],
)

if page == "Release gates":
    st.subheader("Evidence, not claims")
    gates = [
        ("Safety precedence", "Implemented and unit tested"),
        ("Structured LLM + ZDR", "Contract tested; live evaluation pending rotated credential"),
        ("Adaptive policy", "Not implemented; fixed baseline only"),
        ("Episodic memory", "Feature disabled"),
        ("Personalization claim", "Blocked until at least 40 completed benign decisions"),
    ]
    st.dataframe(pd.DataFrame(gates, columns=["Gate", "Current evidence"]), hide_index=True)
elif page == "Simulator":
    st.subheader("Heterogeneous user simulator")
    evidence_table("simulator", "No simulator run exists. Implement Week 5 manually before adding controls.")
elif page == "Policy comparison":
    st.subheader("Regret and policy comparison")
    records = evidence_table(
        "policies", "No policy benchmark exists. Fixed baseline is the only active policy."
    )
    if records and all("step" in row and "cumulative_regret" in row for row in records):
        frame = pd.DataFrame(records)
        st.line_chart(frame, x="step", y="cumulative_regret", color="policy")
elif page == "LLM evaluation":
    st.subheader("Structured extraction candidates")
    st.caption("Promotion requires live, consent-safe runs against the same frozen cases.")
    evidence_table("llm", "No live candidate evaluation artifact exists.")
elif page == "Memory retrieval":
    st.subheader("Retrieval benchmark")
    evidence_table("retrieval", "Memory is disabled and no retrieval benchmark exists.")
elif page == "Safety suite":
    st.subheader("Deterministic safety regression")
    evidence_table("safety", "Run the checked-in safety tests and export measured results here.")
else:
    st.subheader("Experiment registry")
    all_records = []
    for kind in ("simulator", "policies", "llm", "retrieval", "safety"):
        all_records.extend({"kind": kind, **record} for record in load_records(kind))
    if all_records:
        st.dataframe(pd.json_normalize(all_records), width="stretch", hide_index=True)
    else:
        st.info("The registry is empty. It never creates sample evidence automatically.")
