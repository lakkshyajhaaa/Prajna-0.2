"""
utils/hierarchy_viz.py — Prajñā 0.3
Plotly Visualization Helpers for Hierarchical Inference Dashboard

Dark-theme chart builders matching existing 0.2 visualization.py style.
All functions return go.Figure objects ready for st.plotly_chart().
"""

from __future__ import annotations
from typing import Optional
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

# ── Dark theme constants ──────────────────────────────────────────────────
_DARK_BG  = "#0e1117"
_GRID_CLR = "#1f2937"
_FONT_CLR = "#e0e0e0"
_ACCENT   = "#7c3aed"
_GREEN    = "#10b981"
_RED      = "#ef4444"
_AMBER    = "#f59e0b"
_BLUE     = "#3b82f6"


def _dark_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_FONT_CLR, family="Inter, sans-serif"),
        xaxis=dict(gridcolor=_GRID_CLR, zeroline=False),
        yaxis=dict(gridcolor=_GRID_CLR, zeroline=False),
        margin=dict(t=50, b=40, l=40, r=20),
    )
    base.update(kwargs)
    return base


# ── 1. Sankey routing flow ────────────────────────────────────────────────

def plot_routing_path(
    n_gate_reject:  int,
    n_s1_accept:    int,
    n_s1_reject:    int,
    n_s1_escalate:  int,
    n_s2_accept:    int,
    n_s2_reject:    int,
    n_s2_review:    int,
    n_s1_skip:      int = 0,
) -> go.Figure:
    """Sankey diagram showing query volume through each routing path."""
    labels = [
        "Input",        # 0
        "Quality Gate", # 1
        "Hard Reject",  # 2
        "Stage 1",      # 3
        "S1 Accept",    # 4
        "S1 Reject",    # 5
        "Stage 2",      # 6
        "S2 Accept",    # 7
        "S2 Reject",    # 8
        "REVIEW",       # 9
    ]
    total_in  = n_gate_reject + n_s1_accept + n_s1_reject + n_s1_escalate + n_s1_skip
    gate_pass = max(total_in - n_gate_reject, 0)

    raw_links = [
        (0, 1, total_in),
        (1, 2, n_gate_reject),
        (1, 3, gate_pass),
        (3, 4, n_s1_accept),
        (3, 5, n_s1_reject),
        (3, 6, n_s1_escalate + n_s1_skip),
        (6, 7, n_s2_accept),
        (6, 8, n_s2_reject),
        (6, 9, n_s2_review),
    ]
    links = [(s, t, v) for s, t, v in raw_links if v > 0]
    if not links:
        links = [(0, 1, 1)]
    sources, targets, values = zip(*links)

    node_colors = [
        "#6366f1", "#0ea5e9", "#ef4444", "#8b5cf6",
        "#10b981", "#ef4444", "#6366f1", "#10b981",
        "#ef4444", "#f59e0b",
    ]
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=20,
            line=dict(color="#374151", width=0.5),
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=list(sources),
            target=list(targets),
            value=list(values),
            color=["rgba(99,102,241,0.3)"] * len(values),
        ),
    ))
    fig.update_layout(
        title="Routing Flow — Query Volume Through Pipeline Stages",
        **_dark_layout()
    )
    return fig


# ── 2. rho gauge ─────────────────────────────────────────────────────────

def plot_rho_gauge(
    rho:        float,
    rho_accept: float = 0.78,
    rho_reject: float = 0.42,
    stage:      int = 1,
    action:     str = "",
) -> go.Figure:
    """Gauge chart for a single routing score ρ."""
    bar_color = _GREEN if action == "ACCEPT" else _RED if action == "REJECT" else _AMBER

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(rho, 3),
        title={"text": f"Stage {stage} Routing Score (ρ)", "font": {"color": _FONT_CLR}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": _FONT_CLR},
            "bar": {"color": bar_color, "thickness": 0.3},
            "steps": [
                {"range": [0, rho_reject],  "color": "rgba(239,68,68,0.20)"},
                {"range": [rho_reject, rho_accept], "color": "rgba(245,158,11,0.15)"},
                {"range": [rho_accept, 1.0], "color": "rgba(16,185,129,0.20)"},
            ],
        },
        number={"font": {"color": _FONT_CLR}},
    ))
    fig.update_layout(
        paper_bgcolor=_DARK_BG,
        font=dict(color=_FONT_CLR),
        height=260,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


# ── 3. rho distribution histogram ────────────────────────────────────────

def plot_rho_distribution(
    rho_values_s1: list,
    rho_values_s2: Optional[list] = None,
    rho_accept:    float = 0.78,
    rho_reject:    float = 0.42,
) -> go.Figure:
    """Histogram of routing score distribution across evaluated queries."""
    fig = go.Figure()
    if rho_values_s1:
        fig.add_trace(go.Histogram(
            x=rho_values_s1, name="Stage 1 ρ",
            nbinsx=30, opacity=0.75, marker_color=_ACCENT,
        ))
    if rho_values_s2:
        fig.add_trace(go.Histogram(
            x=rho_values_s2, name="Stage 2 ρ",
            nbinsx=30, opacity=0.65, marker_color=_BLUE,
        ))

    fig.add_vrect(x0=0, x1=rho_reject,
                  fillcolor="rgba(239,68,68,0.10)", line_width=0,
                  annotation_text="REJECT", annotation_font_color=_RED)
    fig.add_vrect(x0=rho_reject, x1=rho_accept,
                  fillcolor="rgba(245,158,11,0.10)", line_width=0,
                  annotation_text="ESCALATE", annotation_font_color=_AMBER)
    fig.add_vrect(x0=rho_accept, x1=1.0,
                  fillcolor="rgba(16,185,129,0.10)", line_width=0,
                  annotation_text="ACCEPT", annotation_font_color=_GREEN)

    fig.update_layout(
        title="Routing Score (ρ) Distribution",
        xaxis_title="Routing Score ρ",
        yaxis_title="Query Count",
        barmode="overlay",
        **_dark_layout()
    )
    return fig


# ── 4. Responsibility delta histogram ─────────────────────────────────────

def plot_responsibility_delta(delta_values: list) -> go.Figure:
    """Histogram of ΔR = R2 − R1 for escalated queries."""
    if not delta_values:
        fig = go.Figure()
        fig.update_layout(title="No escalated queries to analyze", **_dark_layout())
        return fig

    deltas    = np.array(delta_values, dtype=float)
    improved  = float((deltas > 0.02).mean() * 100)
    confirmed = float((np.abs(deltas) <= 0.02).mean() * 100)
    degraded  = float((deltas < -0.02).mean() * 100)
    mean_d    = float(np.mean(deltas))

    fig = go.Figure(go.Histogram(
        x=delta_values, nbinsx=25,
        marker_color=_ACCENT, opacity=0.8, name="ΔR = R2 − R1",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=_FONT_CLR,
                  annotation_text="ΔR=0", annotation_font_color=_FONT_CLR)
    fig.add_vline(x=mean_d, line_dash="dot", line_color=_GREEN,
                  annotation_text=f"Mean ΔR={mean_d:.3f}",
                  annotation_font_color=_GREEN)
    fig.update_layout(
        title=(
            f"ΔR Distribution (Escalated Queries) — "
            f"Improved: {improved:.0f}% | Confirmed: {confirmed:.0f}% | Degraded: {degraded:.0f}%"
        ),
        xaxis_title="ΔR = R2 − R1",
        yaxis_title="Query Count",
        **_dark_layout()
    )
    return fig


# ── 5. Stage comparison scatter ───────────────────────────────────────────

def plot_stage_comparison(
    r1_values: list,
    r2_values: list,
    decisions: Optional[list] = None,
) -> go.Figure:
    """Scatter of R1 vs R2 for escalated queries; above diagonal = improvement."""
    if not r1_values or not r2_values:
        fig = go.Figure()
        fig.update_layout(title="No escalated queries to analyze", **_dark_layout())
        return fig

    dec = decisions or ["ACCEPT"] * len(r1_values)
    colors = [_GREEN if d == "ACCEPT" else _RED if d == "REJECT" else _AMBER for d in dec]

    fig = go.Figure(go.Scatter(
        x=r1_values, y=r2_values, mode="markers",
        marker=dict(color=colors, size=8, opacity=0.7,
                    line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
        name="Escalated queries",
        hovertemplate="R1=%{x:.3f}<br>R2=%{y:.3f}<extra></extra>",
    ))
    lo = min(min(r1_values), min(r2_values)) - 0.02
    hi = max(max(r1_values), max(r2_values)) + 0.02
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="rgba(255,255,255,0.2)", dash="dot"),
        name="R1=R2 (no change)",
    ))
    fig.update_layout(
        title="Stage-1 vs Stage-2 Responsibility (Escalated Queries)",
        xaxis_title="Stage-1 Responsibility (R₁)",
        yaxis_title="Stage-2 Responsibility (R₂)",
        **_dark_layout()
    )
    return fig


# ── 6. Compute consumption bar chart ──────────────────────────────────────

def plot_compute_consumption(decisions: list, compute_units: list) -> go.Figure:
    """Average compute units consumed per final decision outcome."""
    if not decisions:
        fig = go.Figure()
        fig.update_layout(title="No data", **_dark_layout())
        return fig

    buckets = defaultdict(list)
    for d, c in zip(decisions, compute_units):
        buckets[d].append(c)

    labels     = list(buckets.keys())
    means      = [float(np.mean(v)) for v in buckets.values()]
    clr_map    = {"ACCEPT": _GREEN, "REJECT": _RED, "REVIEW": _AMBER}
    bar_colors = [clr_map.get(l, _ACCENT) for l in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=means,
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in means],
        textposition="outside",
    ))
    fig.update_layout(
        title="Avg Compute Units by Decision (Stage-2 Only Baseline = 1.0)",
        xaxis_title="Final Decision",
        yaxis_title="Compute Units (relative)",
        **_dark_layout()
    )
    return fig


# ── 7. Decision path text formatter ───────────────────────────────────────

def format_decision_path(
    stages_run:        list,
    terminal_stage:    int,
    routing_action_s1: Optional[str]   = None,
    routing_score_s1:  Optional[float] = None,
    routing_action_s2: Optional[str]   = None,
    routing_score_s2:  Optional[float] = None,
    final_decision:    str = "",
    hard_rejected:     bool = False,
    quality_forced:    bool = False,
) -> str:
    """
    Returns a text-art pipeline path for display in st.code().

    Example:
        Quality Gate → Stage 1 [ρ=0.531 → ESCALATE] → Stage 2 [ρ=0.812 → ACCEPT] → ★ ACCEPT
    """
    parts = ["Quality Gate"]
    if hard_rejected:
        parts.append("HARD REJECT (image quality below minimum)")
        return " → ".join(parts)
    if quality_forced:
        parts.append("[Q forced Stage-2]")
    if 1 in stages_run and routing_action_s1:
        rho_str = f"ρ={routing_score_s1:.3f}" if routing_score_s1 is not None else ""
        parts.append(f"Stage 1 [{rho_str} → {routing_action_s1}]")
    if 2 in stages_run and routing_action_s2:
        rho_str = f"ρ={routing_score_s2:.3f}" if routing_score_s2 is not None else ""
        parts.append(f"Stage 2 [{rho_str} → {routing_action_s2}]")
    if final_decision:
        parts.append(f"★ {final_decision}")
    return " → ".join(parts)


# ── 8. Threshold behavior over queries ────────────────────────────────────

def plot_threshold_behavior(
    query_indices:    list,
    t_accept_values:  list,
    t_review_values:  list,
    rho_values:       Optional[list] = None,
    rho_accept:       float = 0.78,
    rho_reject:       float = 0.42,
) -> go.Figure:
    """Adaptive thresholds and routing scores across query sequence."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=query_indices, y=t_accept_values,
        name="T_accept (adaptive)", mode="lines",
        line=dict(color=_GREEN, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=query_indices, y=t_review_values,
        name="T_review (adaptive)", mode="lines",
        line=dict(color=_AMBER, width=1.5),
    ))
    if rho_values:
        fig.add_trace(go.Scatter(
            x=query_indices, y=rho_values,
            name="ρ routing score", mode="markers",
            marker=dict(color=_ACCENT, size=4, opacity=0.5),
        ))
    fig.add_hline(y=rho_accept, line_dash="dash", line_color=_GREEN,
                  annotation_text=f"ρ_accept={rho_accept}",
                  annotation_font_color=_GREEN)
    fig.add_hline(y=rho_reject, line_dash="dash", line_color=_RED,
                  annotation_text=f"ρ_reject={rho_reject}",
                  annotation_font_color=_RED)
    fig.update_layout(
        title="Adaptive Threshold and Routing Score Behavior Across Queries",
        xaxis_title="Query Index",
        yaxis_title="Score / Threshold Value",
        **_dark_layout()
    )
    return fig


# ── 9. Failure analysis table builder ─────────────────────────────────────

def build_failure_table(pipeline_records: list) -> list:
    """
    Extracts false accept / false reject candidates from pipeline audit records.
    Returns list of dicts suitable for pd.DataFrame display.

    A 'failure candidate' is any record where:
      - decision == ACCEPT but similarity < 0.80 (borderline accept)
      - decision == REJECT but routing_score was in escalate band (missed escalation)
      - decision == REVIEW (all reviews are potential failures)
    """
    rows = []
    for r in pipeline_records:
        decision = r.get("final_decision", "")
        q        = r.get("quality_composite", 0)
        s1       = r.get("stage1") or {}
        s2       = r.get("stage2") or {}

        terminal_stage = r.get("terminal_stage", 0)
        stage_data = s2 if terminal_stage == 2 else s1

        sim = stage_data.get("top1_sim", 0)
        R   = stage_data.get("responsibility", 0)
        rho = stage_data.get("routing_score", 0)

        is_failure_candidate = (
            (decision == "ACCEPT" and sim < 0.80) or
            (decision == "REVIEW") or
            (decision == "REJECT" and rho > 0.35)
        )
        if is_failure_candidate:
            rows.append({
                "Identity":       r.get("predicted_identity", "?"),
                "Decision":       decision,
                "Stage":          terminal_stage,
                "Similarity":     round(sim, 4),
                "R":              round(R, 4),
                "ρ":              round(rho, 4),
                "Quality":        round(q, 4),
                "Stages Run":     str(r.get("stages_run", [])),
                "ΔR":             round(r.get("responsibility_delta") or 0, 4),
                "Review Reasons": "; ".join(r.get("review_reasons", [])),
            })
    return rows
