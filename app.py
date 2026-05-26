"""
app.py — Prajñā 0.3
Responsibility-Guided Hierarchical Inference Framework

5 tabs:
  🔍 Verify        — Hierarchical face verification with routing visualization
  📋 Database      — Dual-stage enrollment + DB viewer
  🔬 Analysis      — 0.2 experiments + 6 hierarchy experiments
  🧭 Pipeline      — Live routing analytics dashboard
  📁 Audit Log     — Combined 0.2 + 0.3 audit log viewer

Backward compatibility: All 0.2 imports and functionality preserved.
The 0.2 Verify workflow is replaced by the hierarchical pipeline.
The 0.2 Analysis experiments remain available.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import os
import json
import time

# ── 0.1 imports (unchanged) ──────────────────────────────────────────────
from frt_utils import (
    calculate_similarity_and_margin,
    compute_entropy_and_certainty,
    compute_quality as compute_quality_v1,
    compute_dynamic_thresholds,
    responsibility_score as responsibility_score_v1,
    final_decision,
)
from llm_utils import generate_responsibility_explanation, get_native_languages

# ── 0.2 imports (unchanged) ──────────────────────────────────────────────
from model_utils import (
    load_models, fetch_and_load_database,
    extract_face_full, load_local_database,
)
from core.metrics import (
    calculate_similarity_and_margin as calc_sim,
    compute_entropy_and_certainty as calc_entropy,
    responsibility_score, compute_ambiguity, compute_open_set_score,
)
from core.quality import compute_composite_quality
from core.thresholds import compute_adaptive_thresholds, explain_threshold
from core.decision import make_decision, STRANGER_FLOOR_DEFAULT
from core.calibration import TAU_CANDIDATES
from utils.logging_utils import (
    AuditRecord, log_decision, load_all_logs, export_csv, export_json,
)
from utils.visualization import (
    plot_roc_curve, plot_far_frr_curve, plot_reliability_diagram,
    plot_entropy_distribution, plot_threshold_distribution,
    plot_weight_comparison, plot_tau_comparison,
)
from utils.language import get_native_languages as get_langs_02

# ── 0.3 imports ───────────────────────────────────────────────────────────
from core.stage1_model import load_stage1_model, get_stage1_model_name
from core.database_manager import (
    load_stage1_database, load_stage2_database,
    enroll_identity as enroll_identity_v3,
    get_database_meta as get_db_meta_v3,
    list_all_identities,
    search_stage1, search_stage2,
)
from core.routing import (
    RHO_ACCEPT_DEFAULT, RHO_REJECT_DEFAULT,
    KAPPA_DEFAULT, LAMBDA_DEFAULT,
)
from core.hierarchy import hierarchical_inference
from utils.logging_utils_v3 import (
    build_pipeline_audit_record, log_pipeline,
    load_all_pipeline_logs, export_pipeline_csv, export_pipeline_json,
)
from utils.hierarchy_viz import (
    plot_rho_gauge, plot_rho_distribution, plot_responsibility_delta,
    plot_stage_comparison, plot_compute_consumption,
    format_decision_path, plot_routing_path, plot_threshold_behavior,
    build_failure_table,
)

# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="प्रज्ञा 0.3 FRT", layout="wide")

st.markdown("""
<h1 style='font-size:48px;'>प्रज्ञा <span style="font-size:22px;color:gray;">FRT</span></h1>
<p style="font-size:20px;">Prajñā 0.3 — Responsibility-Guided Hierarchical Inference</p>
<p style="color:#9ca3af;max-width:900px;">
Two-stage adaptive pipeline: MobileFaceNet → InceptionResnetV1.
Routing score ρ = R · Q<sup>κ</sup> · (1−λA) determines whether to terminate or escalate.
Research-grade governance prototype — not production infrastructure.
</p>
""", unsafe_allow_html=True)

# ── Model + DB loading ────────────────────────────────────────────────────
with st.spinner("Initialising models and databases…"):
    mtcnn, resnet = load_models()
    stage1_state  = load_stage1_model()
    kaggle_db     = fetch_and_load_database(num_classes=None, samples_per_class=1)
    local_db      = load_local_database()
    db_stage1     = load_stage1_database()
    db_stage2_new = load_stage2_database()

# Stage-2 database merges Kaggle + local + new stage2 enrollments
combined_db_s2 = {**kaggle_db, **local_db, **db_stage2_new}

# Stage-1 database: use stage1 enrollments, fallback to stage2 if empty
if not db_stage1:
    db_stage1_effective = {}               # degrade gracefully
    s1_db_empty = True
else:
    db_stage1_effective = db_stage1
    s1_db_empty = False

if not combined_db_s2:
    st.error("No Stage-2 database loaded. Check Kaggle connection or enroll identities.")
    st.stop()

with st.sidebar:
    st.success(f"Kaggle DB: {len(kaggle_db)} identities")
    st.success(f"Local DB (0.2): {len(local_db)} identities")
    st.info(f"Stage-1 DB: {len(db_stage1)} enrolled")
    st.info(f"Stage-2 DB: {len(db_stage2_new)} enrolled (new)")
    st.info(f"Combined S2: {len(combined_db_s2)} identities")
    st.divider()
    st.markdown("**Stage-1 Model:**")
    st.caption(stage1_state.get("model_name", "Unknown"))
    if s1_db_empty:
        st.warning("⚠️ Stage-1 DB empty — hierarchy degraded to Stage-2 only")
    st.divider()
    with st.expander("Routing Parameters"):
        rho_accept = st.slider("ρ_accept", 0.50, 0.95, RHO_ACCEPT_DEFAULT, 0.01,
                               help="Above this: terminate ACCEPT at Stage 1")
        rho_reject = st.slider("ρ_reject", 0.10, 0.60, RHO_REJECT_DEFAULT, 0.01,
                               help="Below this: terminate REJECT at Stage 1")
        kappa      = st.slider("κ (quality exponent)", 0.25, 1.50, KAPPA_DEFAULT, 0.05,
                               help="Higher κ = more aggressive quality penalty")
        lambda_    = st.slider("λ (ambiguity weight)", 0.00, 0.60, LAMBDA_DEFAULT, 0.05,
                               help="Higher λ = stronger ambiguity penalty")
    st.caption("Prajñā 0.3 — Research Prototype")
    st.caption("⚠️ Not production-certified")

# ═════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════
tab_verify, tab_db, tab_analysis, tab_pipeline, tab_logs = st.tabs([
    "🔍 Verify", "📋 Database", "🔬 Analysis", "🧭 Pipeline", "📁 Audit Log"
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — VERIFY (Hierarchical)
# ═══════════════════════════════════════════════════════════════════
with tab_verify:
    st.subheader("Hierarchical Face Verification")
    st.caption(
        "Runs the full 0.3 pipeline: Quality Gate → Stage 1 → Routing → Stage 2 (if escalated). "
        "Stage-1 DB must have enrolled identities for the routing to activate."
    )

    mode_v = st.radio(
        "Mode", ["Single Image", "Batch Analysis"],
        horizontal=True, key="verify_mode"
    )

    if mode_v == "Single Image":
        uploaded = st.file_uploader(
            "Upload face image", type=["jpg", "png", "jpeg"], key="v_upload"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            col1, col2 = st.columns([1, 2])
            with col1:
                thumb = image.copy()
                thumb.thumbnail((400, 400))
                st.image(thumb, caption="Original", width='stretch')

            with st.spinner("Running hierarchical inference…"):
                pr = hierarchical_inference(
                    image=image,
                    mtcnn_model=mtcnn,
                    resnet_model=resnet,
                    db_stage1=db_stage1_effective,
                    db_stage2=combined_db_s2,
                    stage1_model_state=stage1_state,
                    stranger_floor_s1=0.50,
                    stranger_floor_s2=STRANGER_FLOOR_DEFAULT,
                    rho_accept=rho_accept,
                    rho_reject=rho_reject,
                    kappa=kappa,
                    lambda_=lambda_,
                )

            dr = pr.final_decision

            with col2:
                if pr.hard_rejected_at_gate:
                    st.error(f"❌ Gate Reject: {dr.review_reasons[0] if dr.review_reasons else 'Image unprocessable'}")
                else:
                    # ── Decision banner ──────────────────────────────────
                    if dr.decision == "ACCEPT":
                        st.success(f"✅ ACCEPT — Identity: **{dr.predicted_identity}**")
                    elif dr.decision == "REVIEW":
                        st.warning("⚠️ REVIEW — Human verification required")
                    else:
                        st.error("❌ REJECT" + (" (Stranger / non-enrolled)" if dr.is_stranger else ""))

                    # ── Decision Path ────────────────────────────────────
                    st.markdown("#### Decision Path")
                    rr1 = pr.stage1.routing if pr.stage1 else None
                    rr2 = pr.stage2.routing if pr.stage2 else None
                    path_str = format_decision_path(
                        stages_run=pr.stages_run,
                        terminal_stage=pr.terminal_stage,
                        routing_action_s1=rr1.action if rr1 else None,
                        routing_score_s1=rr1.routing_score if rr1 else None,
                        routing_action_s2=rr2.action if rr2 else None,
                        routing_score_s2=rr2.routing_score if rr2 else None,
                        final_decision=dr.decision,
                        hard_rejected=pr.hard_rejected_at_gate,
                        quality_forced=pr.quality_forced_s2,
                    )
                    st.code(path_str, language=None)

                    # ── Routing Score Gauges ────────────────────────────
                    gauge_cols = st.columns(2)
                    with gauge_cols[0]:
                        if pr.stage1 and rr1:
                            st.plotly_chart(
                                plot_rho_gauge(
                                    rho=rr1.routing_score, rho_accept=rho_accept,
                                    rho_reject=rho_reject, stage=1, action=rr1.action
                                ),
                                width='stretch', key="gauge_s1"
                            )
                        else:
                            st.info("Stage-1 not run (DB empty or quality gate reject)")
                    with gauge_cols[1]:
                        if pr.stage2 and rr2:
                            st.plotly_chart(
                                plot_rho_gauge(
                                    rho=rr2.routing_score, rho_accept=rho_accept,
                                    rho_reject=rho_reject, stage=2, action=rr2.action
                                ),
                                width='stretch', key="gauge_s2"
                            )
                        else:
                            if pr.stage2 is None:
                                st.info("Stage-2 not needed (terminated at Stage 1)")

                    # ── Per-Stage Metrics ────────────────────────────────
                    if pr.stage1 or pr.stage2:
                        st.markdown("#### Stage Metrics")
                        stage_rows = []
                        for sm in [pr.stage1, pr.stage2]:
                            if sm is None:
                                continue
                            stage_rows.append({
                                "Stage":       sm.stage,
                                "Model":       sm.model_name,
                                "Top Identity": sm.top1_identity,
                                "Similarity":  round(sm.top1_sim, 4),
                                "R":           round(sm.responsibility, 4),
                                "ρ":           round(sm.routing.routing_score, 4),
                                "Action":      sm.routing.action,
                                "Latency ms":  round(sm.latency_ms, 1),
                            })
                        st.dataframe(pd.DataFrame(stage_rows), width='stretch')

                    # ── Responsibility Evolution ─────────────────────────
                    if pr.responsibility_delta:
                        d = pr.responsibility_delta
                        delta_val = d["delta"]
                        arrow = "⬆" if d["improved"] else ("⬇" if d["degraded"] else "➡")
                        color = "green" if d["improved"] else ("red" if d["degraded"] else "gray")
                        st.markdown(
                            f"**Responsibility Evolution:** "
                            f"R₁={pr.stage1.responsibility:.3f} {arrow} R₂={pr.stage2.responsibility:.3f} "
                            f"(ΔR={delta_val:+.3f})"
                        )

                    # ── Quality ─────────────────────────────────────────
                    if pr.quality:
                        qc = pr.quality
                        with st.expander("Quality Sub-scores"):
                            qdata = {
                                "Component": ["Blur", "Detection Conf", "Brightness", "Face Size", "Pose", "Composite"],
                                "Score":     [qc.blur, qc.confidence, qc.brightness, qc.face_size, qc.pose, qc.composite],
                            }
                            st.dataframe(pd.DataFrame(qdata).round(4), width='stretch')

                    # ── Adaptive Thresholds ──────────────────────────────
                    terminal_sm = pr.stage2 if pr.stage2 else pr.stage1
                    if terminal_sm:
                        with st.expander("Adaptive Thresholds (terminal stage)"):
                            tr = terminal_sm.thresholds
                            th_data = {
                                "Threshold":  ["T_accept (adaptive)", "T_review (adaptive)",
                                               "T_accept (static 0.1)", "T_review (static 0.1)"],
                                "Value":      [tr.t_accept, tr.t_review, 0.72, 0.62],
                            }
                            st.dataframe(pd.DataFrame(th_data).round(4), width='stretch')

                    # ── REVIEW reasons ───────────────────────────────────
                    if dr.decision == "REVIEW" and dr.review_reasons:
                        st.markdown("#### Why REVIEW?")
                        for reason in dr.review_reasons:
                            st.markdown(f"- {reason}")

                    # ── Escalation explanation ───────────────────────────
                    if len(pr.stages_run) > 1:
                        with st.expander("Why was Stage 2 invoked?"):
                            st.info(pr.routing_explanation_s1 or "Stage-1 routing score fell in escalation band.")

                    # ── Compute + Latency ────────────────────────────────
                    c1, c2 = st.columns(2)
                    c1.metric("Compute Units Used", f"{pr.compute_units:.2f}",
                              help="Stage-2-only = 1.0. Stage-1 = 0.15.")
                    c2.metric("Total Latency", f"{pr.total_latency_ms:.0f} ms")

                    # ── Audit log ────────────────────────────────────────
                    audit_rec = build_pipeline_audit_record(
                        pipeline_record=pr,
                        image_filename=uploaded.name,
                        stage1_model_name=stage1_state.get("model_name", "unknown"),
                        rho_accept=rho_accept, rho_reject=rho_reject,
                        kappa=kappa, lambda_=lambda_,
                    )
                    log_pipeline(audit_rec)

                    # ── AI Explanation ────────────────────────────────────
                    st.divider()
                    st.subheader("💡 AI Explanation")
                    langs    = get_native_languages()
                    lang_opts = [f"{k} ({v})" for k, v in langs.items()]
                    sel_full = st.selectbox(
                        "Language", lang_opts,
                        index=lang_opts.index("Hindi (हिन्दी)") if "Hindi (हिन्दी)" in lang_opts else 0,
                        key="v_lang"
                    )
                    sel_lang = sel_full.split(" (")[0]
                    if st.button("✨ Explain Decision", key="v_explain"):
                        active_sm = terminal_sm
                        if active_sm:
                            mdict = {
                                "Top-1 Similarity (S)":     active_sm.top1_sim,
                                "Margin (M)":               active_sm.margin,
                                "Certainty (U)":            active_sm.certainty,
                                "Responsibility Score (R)": active_sm.responsibility,
                                "Routing Score (ρ)":        active_sm.routing.routing_score,
                                "Stages Run":               str(pr.stages_run),
                            }
                            tdict = {
                                "Accept Threshold": active_sm.thresholds.t_accept,
                                "Review Threshold": active_sm.thresholds.t_review,
                            }
                            with st.spinner(f"Generating in {sel_lang}…"):
                                expl = generate_responsibility_explanation(
                                    mdict, dr.decision, tdict, sel_lang
                                )
                                st.info(expl)

    else:  # Batch Analysis
        st.write(
            "Runs all images in `dataset/faces` through the **hierarchical pipeline**."
        )
        if st.button("▶ Start Batch Analysis", key="batch_start"):
            target = "dataset/faces"
            if not os.path.exists(target):
                st.error("dataset/faces not found.")
                st.stop()
            all_imgs = []
            for person in os.listdir(target):
                pdir = os.path.join(target, person)
                if os.path.isdir(pdir):
                    for f in os.listdir(pdir):
                        if f.lower().endswith(("jpg", "jpeg", "png")):
                            all_imgs.append((person, f, os.path.join(pdir, f)))

            if not all_imgs:
                st.warning("No images found.")
            else:
                prog   = st.progress(0)
                status = st.empty()
                results = []
                for idx, (person, fname, fpath) in enumerate(all_imgs):
                    status.text(f"Analysing {idx+1}/{len(all_imgs)}: {person}/{fname}")
                    try:
                        img = Image.open(fpath).convert("RGB")
                        pr_batch = hierarchical_inference(
                            image=img, mtcnn_model=mtcnn, resnet_model=resnet,
                            db_stage1=db_stage1_effective, db_stage2=combined_db_s2,
                            stage1_model_state=stage1_state,
                            stranger_floor_s1=0.50,
                            stranger_floor_s2=STRANGER_FLOOR_DEFAULT,
                            rho_accept=rho_accept, rho_reject=rho_reject,
                            kappa=kappa, lambda_=lambda_,
                        )
                        dr_b = pr_batch.final_decision
                        pred = dr_b.predicted_identity
                        correct = person.split("_")[0] == pred.split("_")[0]
                        results.append({
                            "Image":         fname,
                            "True Identity": person,
                            "Predicted":     pred,
                            "Correct":       "✓" if correct else "✗",
                            "Decision":      dr_b.decision,
                            "Stages Run":    str(pr_batch.stages_run),
                            "R":             round(
                                (pr_batch.stage2 or pr_batch.stage1).responsibility
                                if (pr_batch.stage2 or pr_batch.stage1) else 0, 4
                            ),
                            "ρ":             round(
                                (pr_batch.stage2 or pr_batch.stage1).routing.routing_score
                                if (pr_batch.stage2 or pr_batch.stage1) else 0, 4
                            ),
                            "Compute Units": round(pr_batch.compute_units, 2),
                        })
                    except Exception:
                        pass
                    prog.progress((idx + 1) / len(all_imgs))
                status.text(f"Done — {len(results)} faces processed.")
                if results:
                    df = pd.DataFrame(results)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.subheader("Decision Breakdown")
                        st.bar_chart(df["Decision"].value_counts())
                    with c2:
                        st.subheader("Accuracy")
                        st.bar_chart(df["Correct"].value_counts())
                    with c3:
                        st.subheader("Stages Run")
                        st.bar_chart(df["Stages Run"].value_counts())
                    st.dataframe(df, width='stretch')
                    avg_compute = df["Compute Units"].mean()
                    st.metric("Avg Compute Units", f"{avg_compute:.3f}",
                              delta=f"{(1.0 - avg_compute) * 100:.1f}% vs flat Stage-2",
                              delta_color="inverse")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — DATABASE (Dual-stage enrollment)
# ═══════════════════════════════════════════════════════════════════
with tab_db:
    st.subheader("Identity Database — Dual-Stage Enrollment")
    db_tab1, db_tab2 = st.tabs(["Enroll New Identity", "View Database"])

    with db_tab1:
        st.write(
            "Enrollment creates embeddings in **both** Stage-1 and Stage-2 databases. "
            "Images failing quality validation are logged but not enrolled."
        )
        person_name  = st.text_input("Identity Name", placeholder="e.g. Arun_Kumar")
        uploaded_imgs = st.file_uploader(
            "Upload 1–5 images", type=["jpg", "png", "jpeg"],
            accept_multiple_files=True, key="enroll_imgs"
        )
        if st.button("Enroll Identity", key="enroll_btn"):
            if not person_name.strip():
                st.error("Please enter a name.")
            elif not uploaded_imgs:
                st.error("Please upload at least one image.")
            else:
                pil_imgs = [Image.open(f).convert("RGB") for f in uploaded_imgs]
                with st.spinner(f"Enrolling '{person_name}' in Stage-1 and Stage-2…"):
                    result_enroll = enroll_identity_v3(
                        person_name=person_name.strip(),
                        image_files=pil_imgs,
                        mtcnn_model=mtcnn,
                        resnet_model=resnet,
                        stage1_model_state=stage1_state,
                    )
                if result_enroll["success"]:
                    st.success(result_enroll["message"])
                    st.rerun()
                else:
                    st.error(result_enroll["message"])
                if result_enroll.get("rejection_log"):
                    st.markdown("**Enrollment Log:**")
                    st.dataframe(
                        pd.DataFrame(result_enroll["rejection_log"]),
                        width='stretch'
                    )

    with db_tab2:
        all_ids = list_all_identities()
        st.markdown(f"**Total Identities:** {len(all_ids)}")
        rows = []
        for name, info in all_ids.items():
            meta = info.get("meta", {})
            rows.append({
                "Name":       name,
                "In Stage-1": "✓" if info["in_stage1"] else "✗",
                "In Stage-2": "✓" if info["in_stage2"] else "✗",
                "Source":     meta.get("stage1_model", "Kaggle/Legacy"),
                "Enrolled":   (meta.get("enrolled_at", "")[:10]
                               if meta.get("enrolled_at") else "Kaggle/Legacy"),
            })
        st.dataframe(pd.DataFrame(rows), width='stretch')

        st.markdown("#### Kaggle + Legacy Identities (Stage-2 only)")
        st.dataframe(
            pd.DataFrame({"Identity": sorted(combined_db_s2.keys())}),
            width='stretch'
        )


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab_analysis:
    st.subheader("Experimental Analysis")

    exp_tab_02, exp_tab_03 = st.tabs(["0.2 Experiments (Original)", "0.3 Hierarchy Experiments"])

    # ── 0.2 Experiments (unchanged) ─────────────────────────────────
    with exp_tab_02:
        st.caption(
            "All experiments run on the loaded Stage-2 database. "
            "Genuine pairs = same-identity; Impostor pairs = cross-identity. "
            "Results are dataset-specific — see notes for limitations."
        )
        exp_choice = st.selectbox("Select Experiment", [
            "1 — Adaptive vs Static Thresholds (Primary)",
            "2 — Calibration & Tau Justification",
            "3 — Stranger Rejection Floor",
            "4 — Responsibility Weight Sensitivity",
            "5 — Quality Weight Stability",
        ], key="exp_select")

        if st.button("▶ Run Experiment", key="run_exp"):
            from evaluation.experiments import (
                exp_adaptive_vs_static, exp_calibration_tau,
                exp_stranger_rejection, exp_weight_sensitivity,
                exp_quality_weight_stability,
            )
            with st.spinner("Running experiment…"):
                idx = int(exp_choice[0])
                if idx == 1:
                    res = exp_adaptive_vs_static(combined_db_s2)
                elif idx == 2:
                    res = exp_calibration_tau(combined_db_s2)
                elif idx == 3:
                    res = exp_stranger_rejection(combined_db_s2)
                elif idx == 4:
                    res = exp_weight_sensitivity(combined_db_s2)
                else:
                    res = exp_quality_weight_stability(combined_db_s2)

            if "error" in res:
                st.error(res["error"])
            else:
                import plotly.graph_objects as go

                if idx == 1:
                    ad, st_ = res["adaptive"], res["static"]
                    comp = pd.DataFrame([ad, st_]).set_index("label")[["FAR", "FRR", "TAR", "review_rate"]]
                    st.dataframe(comp.round(4), width='stretch')
                    sig = res["significance"]
                    st.markdown(
                        f"**Paired t-test:** t={sig['t_stat']:.3f}, p={sig['p_value']:.4f} "
                        f"({'statistically significant' if sig['p_value'] < 0.05 else 'not significant at p<0.05'})"
                    )
                    st.info(res["note"])
                    roc = res["roc"]
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"],
                                                  name=f"R-score ROC (AUC={roc['auc']:.3f})",
                                                  line=dict(color="#7c3aed", width=2.5), mode="lines"))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                                  line=dict(color="#555", width=1, dash="dot"),
                                                  mode="lines", showlegend=False))
                    fig_roc.add_trace(go.Scatter(x=[ad["FAR"]], y=[ad["TAR"]],
                                                  name=f"Adaptive (FAR={ad['FAR']:.3f})",
                                                  mode="markers", marker=dict(color="#10b981", size=12, symbol="star")))
                    fig_roc.add_trace(go.Scatter(x=[st_["FAR"]], y=[st_["TAR"]],
                                                  name=f"Static (FAR={st_['FAR']:.3f})",
                                                  mode="markers", marker=dict(color="#ef4444", size=12, symbol="diamond")))
                    fig_roc.update_layout(title="ROC Curve", xaxis_title="FAR", yaxis_title="TAR",
                                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                          font=dict(color="#e0e0e0"))
                    st.plotly_chart(fig_roc, width='stretch')
                    ffr = res["far_frr"]
                    fig_ffr = go.Figure()
                    fig_ffr.add_trace(go.Scatter(x=ffr["thresholds"], y=ffr["far"],
                                                  name="FAR", line=dict(color="#ef4444", width=2)))
                    fig_ffr.add_trace(go.Scatter(x=ffr["thresholds"], y=ffr["frr"],
                                                  name="FRR", line=dict(color="#10b981", width=2)))
                    fig_ffr.update_layout(title="FAR/FRR vs Threshold",
                                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                          font=dict(color="#e0e0e0"))
                    st.plotly_chart(fig_ffr, width='stretch')

                elif idx == 2:
                    tau_res = res["tau_results"]
                    rows = [{"τ": t, "ECE": v["ece"], "Separation": v["separation"],
                              "Mean Genuine H": v["mean_genuine_H"], "Mean Impostor H": v["mean_impostor_H"]}
                             for t, v in tau_res.items()]
                    st.dataframe(pd.DataFrame(rows).round(4), width='stretch')
                    st.info(res["note"])
                    tau_vals = list(tau_res.keys())
                    st.plotly_chart(plot_tau_comparison(
                        tau_vals, [tau_res[t]["ece"] for t in tau_vals],
                        [tau_res[t]["separation"] for t in tau_vals], chosen_tau=0.03
                    ), width='stretch')
                    chosen = tau_res.get(0.03, list(tau_res.values())[0])
                    st.plotly_chart(plot_reliability_diagram(
                        chosen["bin_confs"], chosen["bin_accs"], [], chosen["ece"], tau=0.03
                    ), width='stretch')

                elif idx == 3:
                    fr = pd.DataFrame(res["floor_results"])
                    st.dataframe(fr.round(4), width='stretch')
                    st.info(res["note"])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fr["floor"], y=fr["TAR"], name="TAR", line=dict(color="#10b981")))
                    fig.add_trace(go.Scatter(x=fr["floor"], y=fr["FAR"], name="FAR", line=dict(color="#ef4444")))
                    fig.add_vline(x=res["optimal_floor"], line_dash="dash", line_color="#facc15",
                                  annotation_text=f"Optimal floor={res['optimal_floor']}")
                    fig.update_layout(title="TAR/FAR vs Stranger Rejection Floor",
                                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                      font=dict(color="#e0e0e0"))
                    st.plotly_chart(fig, width='stretch')

                elif idx == 4:
                    rows4 = [{"Weights": r["label"], "FAR": r["FAR"], "FRR": r["FRR"],
                               "TAR": r["TAR"], "R Variance": r["r_variance"]} for r in res["results"]]
                    st.dataframe(pd.DataFrame(rows4).round(4), width='stretch')
                    st.info(res["note"])
                    st.plotly_chart(plot_weight_comparison(
                        [r["label"] for r in res["results"]],
                        [r["FAR"] for r in res["results"]],
                        [r["FRR"] for r in res["results"]],
                        [r["r_variance"] for r in res["results"]],
                    ), width='stretch')

                else:
                    st.markdown(f"**FAR range:** {res['far_range']:.4f} | **FRR range:** {res['frr_range']:.4f}")
                    st.info(res["note"])
                    fig5 = go.Figure()
                    fig5.add_trace(go.Bar(x=res["weight_labels"], y=res["far_values"],
                                          name="FAR", marker_color="#ef4444"))
                    fig5.add_trace(go.Bar(x=res["weight_labels"], y=res["frr_values"],
                                          name="FRR", marker_color="#10b981"))
                    fig5.update_layout(title="FAR/FRR Across 10 Random Quality Weight Sets",
                                       barmode="group", paper_bgcolor="#0e1117",
                                       plot_bgcolor="#0e1117", font=dict(color="#e0e0e0"))
                    st.plotly_chart(fig5, width='stretch')

    # ── 0.3 Hierarchy Experiments ────────────────────────────────────
    with exp_tab_03:
        st.caption(
            "Hierarchy experiments run on Stage-1 and Stage-2 enrollment databases. "
            "Stage-1 DB must have enrolled identities. "
            "Results include explicit limitation notes."
        )
        hier_exp_choice = st.selectbox("Select Experiment", [
            "1 — Single-Stage vs Hierarchical",
            "2 — Compute Reduction Analysis",
            "3 — Latency Analysis",
            "4 — False Accept Recovery",
            "5 — Review Rate Comparison",
            "6 — Routing Ablation Study",
        ], key="hier_exp_select")

        if st.button("▶ Run Hierarchy Experiment", key="run_hier_exp"):
            from evaluation.exp_hierarchy import (
                exp_single_vs_hierarchical,
                exp_compute_reduction,
                exp_latency_analysis,
                exp_false_accept_recovery,
                exp_review_rate_comparison,
                exp_routing_ablation,
            )
            db_s1_eval = db_stage1 if db_stage1 else combined_db_s2
            db_s2_eval = combined_db_s2

            with st.spinner("Running hierarchy experiment…"):
                idx_h = int(hier_exp_choice[0])
                if idx_h == 1:
                    res_h = exp_single_vs_hierarchical(db_s1_eval, db_s2_eval,
                                                        rho_accept=rho_accept, rho_reject=rho_reject,
                                                        kappa=kappa, lambda_=lambda_)
                elif idx_h == 2:
                    res_h = exp_compute_reduction(db_s1_eval, rho_accept=rho_accept,
                                                   rho_reject=rho_reject, kappa=kappa, lambda_=lambda_)
                elif idx_h == 3:
                    res_h = exp_latency_analysis(db_s1_eval, db_s2_eval)
                elif idx_h == 4:
                    res_h = exp_false_accept_recovery(db_s1_eval, db_s2_eval,
                                                       rho_accept=rho_accept, rho_reject=rho_reject,
                                                       kappa=kappa, lambda_=lambda_)
                elif idx_h == 5:
                    res_h = exp_review_rate_comparison(db_s1_eval, db_s2_eval,
                                                        rho_accept=rho_accept, rho_reject=rho_reject,
                                                        kappa=kappa, lambda_=lambda_)
                else:
                    res_h = exp_routing_ablation(db_s1_eval, rho_accept=rho_accept, rho_reject=rho_reject)

            if "error" in res_h:
                st.error(res_h["error"])
            else:
                import plotly.graph_objects as go

                if idx_h == 1:
                    s1_m = res_h["flat_s1"]
                    s2_m = res_h["flat_s2"]
                    h_m  = res_h["hierarchical"]
                    valid_rows = [m for m in [s1_m, s2_m, h_m] if "FAR" in m and m["FAR"] is not None]
                    if valid_rows:
                        st.dataframe(pd.DataFrame(valid_rows).set_index("label").round(4),
                                     width='stretch')
                    if "compute_savings_pct" in res_h:
                        st.metric("Compute Savings vs Flat Stage-2",
                                  f"{res_h['compute_savings_pct']:.1f}%")
                    st.info(res_h["note"])
                    st.warning(res_h["limitation"])

                elif idx_h == 2:
                    dr_rows = res_h["decile_results"]
                    st.dataframe(pd.DataFrame(dr_rows).round(4), width='stretch')
                    st.info(res_h["note"])
                    if dr_rows:
                        fig_cr = go.Figure()
                        fig_cr.add_trace(go.Scatter(
                            x=[r["Q_level"] for r in dr_rows],
                            y=[r["f1_termination_rate"] for r in dr_rows],
                            name="Stage-1 Termination Rate",
                            line=dict(color="#7c3aed", width=2), mode="lines+markers"
                        ))
                        fig_cr.update_layout(
                            title="Stage-1 Termination Rate by Quality Level",
                            xaxis_title="Quality Q", yaxis_title="f1 Termination Rate",
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            font=dict(color="#e0e0e0")
                        )
                        st.plotly_chart(fig_cr, width='stretch')
                    if res_h.get("rho_values"):
                        st.plotly_chart(
                            plot_rho_distribution(res_h["rho_values"], rho_accept=rho_accept,
                                                   rho_reject=rho_reject),
                            width='stretch'
                        )

                elif idx_h == 3:
                    s1_lat = res_h["stage1_search"]
                    s2_lat = res_h["stage2_search"]
                    lat_rows = [
                        {"Stage": "Stage 1", "Median ms": s1_lat["median_ms"],
                         "p95 ms": s1_lat["p95_ms"], "DB Size": s1_lat["db_size"]},
                        {"Stage": "Stage 2", "Median ms": s2_lat["median_ms"],
                         "p95 ms": s2_lat["p95_ms"], "DB Size": s2_lat["db_size"]},
                    ]
                    st.dataframe(pd.DataFrame(lat_rows), width='stretch')
                    st.metric("Breakeven f1 Rate", f"{res_h['f1_breakeven']:.3f}",
                              help="Hierarchy faster only when >f1_breakeven queries terminate at Stage 1")
                    st.info(res_h["note"])
                    if res_h.get("f1_sweep") and res_h.get("expected_latency_ms"):
                        fig_lat = go.Figure()
                        fig_lat.add_trace(go.Scatter(
                            x=res_h["f1_sweep"], y=res_h["expected_latency_ms"],
                            name="Hierarchical Latency", line=dict(color="#7c3aed", width=2)
                        ))
                        fig_lat.add_hline(y=res_h["flat_s2_latency_ms"], line_dash="dash",
                                          line_color="#ef4444",
                                          annotation_text=f"Flat S2 = {res_h['flat_s2_latency_ms']:.1f}ms",
                                          annotation_font_color="#ef4444")
                        fig_lat.update_layout(
                            title="Expected Latency vs Stage-1 Termination Rate",
                            xaxis_title="f1 Termination Rate", yaxis_title="Latency (ms)",
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            font=dict(color="#e0e0e0")
                        )
                        st.plotly_chart(fig_lat, width='stretch')

                elif idx_h == 4:
                    st.metric("Stage-1 False Accepts", res_h["n_fa_stage1"])
                    st.metric("Recovered by Stage-2", res_h["n_recovered"],
                              delta=f"{res_h['recovery_rate']*100:.1f}% recovery rate")
                    st.info(res_h["note"])
                    st.warning(res_h["limitation"])
                    if res_h.get("recovery_details"):
                        st.dataframe(pd.DataFrame(res_h["recovery_details"]).round(4),
                                     width='stretch')

                elif idx_h == 5:
                    review_rows = [
                        {"Condition":    "Flat Stage-2",
                         "Review Rate":  res_h["flat_s2_review_rate"],
                         "Review (genuine)":  res_h["flat_review_genuine"],
                         "Review (impostor)": res_h["flat_review_impostor"]},
                        {"Condition":    "Hierarchical",
                         "Review Rate":  res_h["hierarchical_review_rate"],
                         "Review (genuine)":  None,
                         "Review (impostor)": None},
                    ]
                    st.dataframe(pd.DataFrame(review_rows).round(4), width='stretch')
                    st.metric("Review Reduction", f"{res_h['review_reduction_pct']:.2f}%")
                    st.info(res_h["note"])

                else:  # Ablation
                    st.dataframe(pd.DataFrame(res_h["ablation_results"]).round(4),
                                 width='stretch')
                    st.info(res_h["note"])
                    st.warning(res_h["limitation"])
                    if res_h.get("kappa_sweep"):
                        kappa_df = pd.DataFrame(res_h["kappa_sweep"])
                        st.subheader("κ Sweep")
                        st.dataframe(kappa_df.round(4), width='stretch')


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — PIPELINE ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.subheader("Pipeline Analytics Dashboard")
    st.caption("Live analytics computed from pipeline audit logs (logs/pipeline_*.jsonl).")

    pipeline_records = load_all_pipeline_logs()

    if not pipeline_records:
        st.info(
            "No pipeline records yet. Run verifications in the Verify tab to generate data. "
            "Each verification writes one record to logs/pipeline_YYYYMMDD.jsonl."
        )
    else:
        df_pipe = pd.DataFrame(pipeline_records)

        # ── Summary metrics ───────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Queries",  len(df_pipe))
        if "final_decision" in df_pipe.columns:
            col2.metric("Accepts",  int((df_pipe["final_decision"] == "ACCEPT").sum()))
            col3.metric("Reviews",  int((df_pipe["final_decision"] == "REVIEW").sum()))
            col4.metric("Rejects",  int((df_pipe["final_decision"] == "REJECT").sum()))
        if "compute_units" in df_pipe.columns:
            col5.metric("Avg Compute", f"{df_pipe['compute_units'].mean():.3f}")

        st.divider()

        # ── Routing Sankey ─────────────────────────────────────────────
        st.subheader("Routing Flow")
        if "stages_run" in df_pipe.columns:
            def _count(cond):
                return int(cond.sum()) if hasattr(cond, "sum") else 0

            hard_rej = _count(df_pipe.get("hard_rejected_at_gate", pd.Series([False]*len(df_pipe))))
            runs_s1  = df_pipe["stages_run"].apply(lambda x: 1 in (x or []))
            runs_s2  = df_pipe["stages_run"].apply(lambda x: 2 in (x or []))

            # Stage-1 actions
            s1_accept  = 0; s1_reject = 0; s1_escalate = 0
            for rec in pipeline_records:
                s1 = rec.get("stage1")
                if s1:
                    act = s1.get("routing_action", "")
                    if act == "ACCEPT":   s1_accept  += 1
                    elif act == "REJECT": s1_reject  += 1
                    elif act == "ESCALATE": s1_escalate += 1

            s2_accept  = _count((df_pipe["final_decision"] == "ACCEPT") & runs_s2)
            s2_reject  = _count((df_pipe["final_decision"] == "REJECT") & runs_s2)
            s2_review  = _count((df_pipe["final_decision"] == "REVIEW") & runs_s2)
            s1_skip    = _count(runs_s2 & ~runs_s1)

            st.plotly_chart(plot_routing_path(
                n_gate_reject=hard_rej,
                n_s1_accept=s1_accept, n_s1_reject=s1_reject,
                n_s1_escalate=s1_escalate, n_s2_accept=s2_accept,
                n_s2_reject=s2_reject, n_s2_review=s2_review,
                n_s1_skip=s1_skip,
            ), width='stretch')

        st.divider()

        # ── rho distribution ───────────────────────────────────────────
        st.subheader("Routing Score Distribution")
        rho_s1_vals = [rec["stage1"]["routing_score"] for rec in pipeline_records
                       if rec.get("stage1") and rec["stage1"]]
        rho_s2_vals = [rec["stage2"]["routing_score"] for rec in pipeline_records
                       if rec.get("stage2") and rec["stage2"]]
        if rho_s1_vals or rho_s2_vals:
            st.plotly_chart(plot_rho_distribution(
                rho_s1_vals, rho_s2_vals, rho_accept=rho_accept, rho_reject=rho_reject
            ), width='stretch')

        st.divider()

        # ── Responsibility Delta ───────────────────────────────────────
        st.subheader("Responsibility Evolution (ΔR)")
        delta_vals = [r["responsibility_delta"] for r in pipeline_records
                      if r.get("responsibility_delta") is not None]
        if delta_vals:
            st.plotly_chart(plot_responsibility_delta(delta_vals), width='stretch')
        else:
            st.info("No escalated queries with ΔR data yet.")

        st.divider()

        # ── Stage comparison scatter ────────────────────────────────────
        st.subheader("Stage-1 vs Stage-2 Responsibility")
        r1_vals, r2_vals, dec_vals = [], [], []
        for rec in pipeline_records:
            if rec.get("stage1") and rec.get("stage2"):
                r1_vals.append(rec["stage1"].get("responsibility", 0))
                r2_vals.append(rec["stage2"].get("responsibility", 0))
                dec_vals.append(rec.get("final_decision", "ACCEPT"))
        if r1_vals:
            st.plotly_chart(plot_stage_comparison(r1_vals, r2_vals, dec_vals),
                            width='stretch')
        else:
            st.info("No escalated queries (Stage-1 + Stage-2) to plot yet.")

        st.divider()

        # ── Compute consumption ─────────────────────────────────────────
        st.subheader("Compute Consumption")
        if "final_decision" in df_pipe.columns and "compute_units" in df_pipe.columns:
            st.plotly_chart(plot_compute_consumption(
                decisions=df_pipe["final_decision"].tolist(),
                compute_units=df_pipe["compute_units"].tolist(),
            ), width='stretch')

        st.divider()

        # ── Failure analysis ────────────────────────────────────────────
        st.subheader("Failure Analysis Candidates")
        fail_rows = build_failure_table(pipeline_records)
        if fail_rows:
            st.dataframe(pd.DataFrame(fail_rows).round(4), width='stretch')
        else:
            st.success("No failure candidates in current audit log.")

        st.divider()

        # ── Export ─────────────────────────────────────────────────────
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            csv_p = export_pipeline_csv(pipeline_records)
            st.download_button("⬇ Export Pipeline CSV", csv_p,
                               "prajna_pipeline.csv", "text/csv")
        with col_e2:
            json_p = export_pipeline_json(pipeline_records)
            st.download_button("⬇ Export Pipeline JSON", json_p,
                               "prajna_pipeline.json", "application/json")


# ═══════════════════════════════════════════════════════════════════
# TAB 5 — AUDIT LOG (Combined 0.2 + 0.3)
# ═══════════════════════════════════════════════════════════════════
with tab_logs:
    st.subheader("Audit Log")
    log_tab1, log_tab2 = st.tabs(["0.2 Decision Log", "0.3 Pipeline Log"])

    with log_tab1:
        records = load_all_logs()
        if not records:
            st.info("No 0.2 audit records. Run a verification to generate entries.")
        else:
            df_log = pd.DataFrame(records)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Decisions", len(df_log))
            if "decision" in df_log.columns:
                c2.metric("Accepts", int((df_log["decision"] == "ACCEPT").sum()))
                c3.metric("Reviews", int((df_log["decision"] == "REVIEW").sum()))
            cols_show = [c for c in ["timestamp", "predicted_identity", "decision", "similarity",
                                      "entropy", "quality_composite", "t_accept_used",
                                      "responsibility_score", "is_stranger_flag"] if c in df_log.columns]
            st.dataframe(df_log[cols_show].round(4), width='stretch')
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.download_button("⬇ Export CSV", export_csv(records),
                                   "prajna_audit_02.csv", "text/csv")
            with col_e2:
                st.download_button("⬇ Export JSON", export_json(records),
                                   "prajna_audit_02.json", "application/json")

    with log_tab2:
        p_records = load_all_pipeline_logs()
        if not p_records:
            st.info("No 0.3 pipeline records. Run verifications in the Verify tab.")
        else:
            df_p = pd.DataFrame(p_records)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Pipeline Runs", len(df_p))
            if "final_decision" in df_p.columns:
                c2.metric("Accepts", int((df_p["final_decision"] == "ACCEPT").sum()))
                c3.metric("Reviews", int((df_p["final_decision"] == "REVIEW").sum()))

            cols_p = [c for c in ["timestamp", "predicted_identity", "final_decision",
                                   "stages_run", "terminal_stage", "compute_units",
                                   "total_latency_ms", "quality_composite",
                                   "responsibility_delta", "is_stranger"] if c in df_p.columns]
            st.dataframe(df_p[cols_p].round(4), width='stretch')

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.download_button("⬇ Export CSV", export_pipeline_csv(p_records),
                                   "prajna_pipeline.csv", "text/csv")
            with col_e2:
                st.download_button("⬇ Export JSON", export_pipeline_json(p_records),
                                   "prajna_pipeline.json", "application/json")
