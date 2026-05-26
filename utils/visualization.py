"""
utils/visualization.py — Prajñā 0.2
Shared Plotly chart builders for the Streamlit Analysis tab.

All charts use a consistent dark theme matching the existing Streamlit UI.
Each function returns a Plotly Figure object — Streamlit renders via st.plotly_chart().

Functions:
  - plot_roc_curve            — ROC + AUC for adaptive vs static
  - plot_far_frr_curve        — FAR/FRR vs threshold (EER visualization)
  - plot_reliability_diagram  — Calibration check (accuracy vs confidence)
  - plot_entropy_distribution — Genuine vs impostor entropy separation
  - plot_threshold_distribution — Per-query adaptive threshold variance
  - plot_weight_comparison    — R-score variance across weight combinations
  - plot_tau_comparison       — ECE and entropy stats across tau values
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Shared dark theme config
# ---------------------------------------------------------------------------
DARK_LAYOUT = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#e0e0e0", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#2d2d2d", zeroline=False),
    yaxis=dict(gridcolor="#2d2d2d", zeroline=False),
    legend=dict(bgcolor="#1a1a2e", bordercolor="#3d3d5c"),
    margin=dict(l=50, r=30, t=60, b=50),
)

ACCENT_ADAPTIVE = "#7c3aed"   # purple — adaptive system
ACCENT_STATIC   = "#ef4444"   # red — static baseline
ACCENT_GENUINE  = "#10b981"   # green — genuine / correct
ACCENT_IMPOSTOR = "#f59e0b"   # amber — impostor / incorrect


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    fpr_adaptive: list,
    tpr_adaptive: list,
    auc_adaptive: float,
    fpr_static: list,
    tpr_static: list,
    auc_static: float,
) -> go.Figure:
    """ROC curves comparing adaptive vs static thresholds."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr_adaptive, y=tpr_adaptive,
        name=f"Adaptive (AUC={auc_adaptive:.3f})",
        line=dict(color=ACCENT_ADAPTIVE, width=2.5),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=fpr_static, y=tpr_static,
        name=f"Static 0.1 baseline (AUC={auc_static:.3f})",
        line=dict(color=ACCENT_STATIC, width=2, dash="dash"),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random classifier",
        line=dict(color="#555", width=1, dash="dot"),
        mode="lines",
        showlegend=False,
    ))
    fig.update_layout(
        title="ROC Curve — Adaptive vs Static Thresholds",
        xaxis_title="False Accept Rate (FAR)",
        yaxis_title="True Accept Rate (TAR)",
        **DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# FAR / FRR Curve
# ---------------------------------------------------------------------------

def plot_far_frr_curve(
    thresholds: list,
    far_adaptive: list,
    frr_adaptive: list,
    far_static: list,
    frr_static: list,
) -> go.Figure:
    """FAR and FRR vs threshold, showing EER points for both systems."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=far_adaptive,
        name="FAR — Adaptive",
        line=dict(color=ACCENT_ADAPTIVE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=frr_adaptive,
        name="FRR — Adaptive",
        line=dict(color=ACCENT_ADAPTIVE, width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=far_static,
        name="FAR — Static",
        line=dict(color=ACCENT_STATIC, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=frr_static,
        name="FRR — Static",
        line=dict(color=ACCENT_STATIC, width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title="FAR / FRR vs Threshold",
        xaxis_title="Threshold Value",
        yaxis_title="Error Rate",
        **DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Reliability Diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    bin_confs: list,
    bin_accs: list,
    bin_counts: list,
    ece: float,
    tau: float = 0.03,
) -> go.Figure:
    """
    Reliability diagram (calibration curve).
    Diagonal = perfect calibration. Below = overconfident. Above = underconfident.
    """
    fig = go.Figure()

    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Perfect calibration",
        line=dict(color="#555", width=1.5, dash="dot"),
        mode="lines",
    ))

    # Actual calibration curve
    fig.add_trace(go.Scatter(
        x=bin_confs, y=bin_accs,
        name=f"Model (ECE={ece:.4f})",
        line=dict(color=ACCENT_ADAPTIVE, width=2.5),
        mode="lines+markers",
        marker=dict(size=8),
    ))

    # Gap fill (overconfidence region)
    fig.add_trace(go.Scatter(
        x=bin_confs + bin_confs[::-1],
        y=bin_confs + bin_accs[::-1],
        fill="toself",
        fillcolor="rgba(124, 58, 237, 0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Calibration gap",
        showlegend=True,
    ))

    fig.update_layout(
        title=f"Reliability Diagram — τ={tau} (ECE={ece:.4f})",
        xaxis_title="Confidence (R Score)",
        yaxis_title="Fraction Correct",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        **DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Entropy Distribution
# ---------------------------------------------------------------------------

def plot_entropy_distribution(
    genuine_H: list,
    impostor_H: list,
    tau: float,
    separation: float,
) -> go.Figure:
    """
    Overlapping histograms of entropy for genuine and impostor queries.
    Good calibration: genuine H peaks left (low), impostor H peaks right (high).
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=genuine_H,
        name="Genuine queries",
        marker_color=ACCENT_GENUINE,
        opacity=0.7,
        nbinsx=30,
    ))
    fig.add_trace(go.Histogram(
        x=impostor_H,
        name="Impostor / stranger queries",
        marker_color=ACCENT_IMPOSTOR,
        opacity=0.7,
        nbinsx=30,
    ))
    fig.update_layout(
        barmode="overlay",
        title=f"Entropy Distribution — τ={tau} (separation={separation:.3f})",
        xaxis_title="Shannon Entropy H",
        yaxis_title="Count",
        **DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Threshold Distribution (per-query adaptive variation)
# ---------------------------------------------------------------------------

def plot_threshold_distribution(
    adaptive_t_accepts: list,
    static_t_accept: float = 0.72,
) -> go.Figure:
    """
    Histogram of per-query adaptive T_accept values.
    The static baseline is shown as a vertical line.
    Shows the system is NOT using a single fixed threshold.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=adaptive_t_accepts,
        name="Adaptive T_accept (per query)",
        marker_color=ACCENT_ADAPTIVE,
        opacity=0.8,
        nbinsx=40,
    ))
    fig.add_vline(
        x=static_t_accept,
        line_dash="dash",
        line_color=ACCENT_STATIC,
        annotation_text=f"Static baseline ({static_t_accept})",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Adaptive Threshold Distribution Across Queries",
        xaxis_title="T_accept Value",
        yaxis_title="Query Count",
        **DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Weight Comparison
# ---------------------------------------------------------------------------

def plot_weight_comparison(
    weight_labels: list[str],
    far_values: list[float],
    frr_values: list[float],
    r_variances: list[float],
) -> go.Figure:
    """
    Grouped bar chart comparing FAR, FRR, and R-score variance across
    responsibility weight combinations.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["FAR & FRR by Weight Combination", "R-Score Variance"],
    )

    fig.add_trace(go.Bar(
        name="FAR", x=weight_labels, y=far_values,
        marker_color=ACCENT_STATIC, opacity=0.85,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        name="FRR", x=weight_labels, y=frr_values,
        marker_color=ACCENT_GENUINE, opacity=0.85,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        name="R Variance", x=weight_labels, y=r_variances,
        marker_color=ACCENT_ADAPTIVE, opacity=0.85,
    ), row=1, col=2)

    fig.update_layout(
        title="Responsibility Weight Sensitivity Analysis",
        barmode="group",
        **DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Tau Comparison (ECE + entropy separation)
# ---------------------------------------------------------------------------

def plot_tau_comparison(
    tau_values: list[float],
    ece_values: list[float],
    separation_values: list[float],
    chosen_tau: float = 0.03,
) -> go.Figure:
    """
    Dual-axis plot: ECE (lower = better) and entropy separation (higher = better)
    across tau values. The chosen tau should minimize ECE while maintaining separation.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    tau_str = [str(t) for t in tau_values]

    fig.add_trace(go.Scatter(
        x=tau_str, y=ece_values,
        name="ECE (lower = better)",
        line=dict(color=ACCENT_STATIC, width=2.5),
        mode="lines+markers",
        marker=dict(size=9),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=tau_str, y=separation_values,
        name="Entropy Separation (higher = better)",
        line=dict(color=ACCENT_GENUINE, width=2.5),
        mode="lines+markers",
        marker=dict(size=9),
    ), secondary_y=True)

    # Mark chosen tau
    if str(chosen_tau) in tau_str:
        idx = tau_str.index(str(chosen_tau))
        fig.add_vline(
            x=idx,
            line_dash="dash",
            line_color="#facc15",
            annotation_text=f"Chosen τ={chosen_tau}",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Temperature (τ) Sensitivity — ECE vs Entropy Separation",
        **DARK_LAYOUT,
    )
    fig.update_xaxes(title_text="Softmax Temperature τ")
    fig.update_yaxes(title_text="ECE", secondary_y=False)
    fig.update_yaxes(title_text="Entropy Separation", secondary_y=True)
    return fig
