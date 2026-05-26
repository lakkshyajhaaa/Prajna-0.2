# प्रज्ञा (Prajñā) 0.3
## Responsibility-Guided Hierarchical Inference Framework

> *"AI should not only make predictions. It should decide whether its predictions deserve to be trusted — and how much intelligence to spend before deciding."*

---

## What is Prajñā 0.3?

Prajñā 0.3 is a research-grade, governance-first face recognition system. It extends the adaptive uncertainty-awareness of 0.2 with a **two-stage hierarchical inference pipeline** driven by a routing score that determines how much compute to invest in each decision.

The key question the system answers at every query:
> *"Is Stage-1's confidence sufficient to decide, or should I spend Stage-2 compute?"*

---

## Architecture

```
                       PIL Image
                           │
                    ┌──────▼──────┐
                    │ Quality Gate │   Q < 0.20 → Hard REJECT
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │         Stage 1          │
              │  MobileFaceNet (ONNX)    │
              │  or IRESNET-112px fallback│
              │                          │
              │  Routing Score:          │
              │  ρ = R · Q^κ · (1−λA)   │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    Routing Decision      │
              │  ρ ≥ 0.78 → ACCEPT      │
              │  ρ ≤ 0.42 → REJECT      │
              │  else → ESCALATE         │
              └─────┬──────────┬────────┘
              ACCEPT│          │ESCALATE/REJECT
              ─────▼          ▼──────────────
           Terminal        ┌──────────────────┐
           Stage-1         │     Stage 2       │
                           │ InceptionResnetV1 │
                           │ /VGGFace2 (512-d) │
                           └────────┬──────────┘
                                    │
                           ┌────────▼──────────┐
                           │  Final Decision    │
                           │ ACCEPT/REVIEW/REJECT│
                           └───────────────────┘
```

### Routing Score Formula

```
ρ = R · φ(Q) · ψ(A)

where:
  R      = responsibility score (cosine similarity weighted by margin + certainty)
  φ(Q)   = Q^κ    (quality attenuation, default κ=0.50)
  ψ(A)   = 1−λ·A  (ambiguity attenuation, default λ=0.30)
  Q      = composite image quality score ∈ [0,1]
  A      = top-2/top-1 similarity ratio (identity ambiguity) ∈ [0,1]
```

Routing thresholds:
- `ρ_accept = 0.78` — terminate with ACCEPT (configurable in sidebar)
- `ρ_reject = 0.42` — terminate with REJECT

---

## Version History

| Version | Key Addition |
|---------|-------------|
| 0.1 | MTCNN + InceptionResnetV1, static thresholds, open-set rejection, responsibility R |
| 0.2 | Adaptive thresholds (entropy + quality + ambiguity), composite quality scoring, enrollment DB, calibration analysis, multilingual explanations |
| **0.3** | **Two-stage hierarchy, routing score ρ, Stage-1 MobileFaceNet, dual DB, compute savings accounting, 6 new experiments, pipeline analytics dashboard** |

---

## New in 0.3

- **`core/stage1_model.py`** — MobileFaceNet ONNX wrapper with InceptionResnetV1 fallback
- **`core/database_manager.py`** — Dual-stage enrollment DB (`database/stage1/` + `database/stage2/`)
- **`core/routing.py`** — Routing score ρ formula, routing decisions, `RoutingRecord`
- **`core/hierarchy.py`** — `hierarchical_inference()` orchestrator, `PipelineRecord`
- **`utils/logging_utils_v3.py`** — Extended audit logging with per-stage metrics
- **`utils/hierarchy_viz.py`** — Sankey routing flow, ρ gauges, ΔR histograms, stage scatter
- **`evaluation/exp_hierarchy.py`** — 6 hierarchy experiments
- **`tests/test_routing.py`** — 23 routing engine tests (all passing)
- **`tests/test_database_manager.py`** — 9 database manager tests (all passing)

---

## Streamlit Tabs

| Tab | Description |
|-----|-------------|
| 🔍 Verify | Hierarchical face verification with routing gauges, decision path, stage metrics |
| 📋 Database | Dual-stage identity enrollment + database viewer |
| 🔬 Analysis | 5 original 0.2 experiments + 6 new hierarchy experiments |
| 🧭 Pipeline | Live routing analytics: Sankey flow, ρ distribution, ΔR histogram, compute chart |
| 📁 Audit Log | Combined 0.2 decision log + 0.3 pipeline log with CSV/JSON export |

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

To use ONNX MobileFaceNet (optional — faster Stage-1):
1. Download a MobileFaceNet-512 ONNX checkpoint
2. Place at `models/mobilefacenet.onnx`
3. Set `MOBILEFACENET_ONNX_PATH` env var if using a custom path

Without the ONNX model, the system automatically uses InceptionResnetV1 at 112px as Stage-1 fallback.

---

## Governance Philosophy

The routing score ρ is the system's answer to:
> *"Does this stage's evidence justify a terminal decision, or should I invest more compute?"*

Key design principles:
- **No opaque decisions** — every routing decision has a logged derivation
- **Explainable escalation** — ESCALATE always comes with human-readable reasons
- **Compute accountability** — every decision records `compute_units` (Stage-2 = 1.0)
- **Graceful degradation** — Stage-1 DB empty → degrades to Stage-2-only seamlessly
- **Open-set rejection preserved** — stranger floor from 0.1/0.2 is enforced at every stage

---

## Limitations (Documented)

| Limitation | Mitigation |
|-----------|------------|
| Routing thresholds (ρ_accept, ρ_reject) are calibrated heuristically | Experiment 6 ablation sweep; recalibrate on held-out data |
| Stage-1 and Stage-2 similarity scores are not directly comparable | Each stage queries its own embedding DB; ΔR is an approximation |
| Experiment Q scores use placeholder (0.70) when only embeddings stored | Acknowledged in every experiment note field |
| MobileFaceNet fallback (112px IRESNET) is not a true Stage-1 model | Users can supply a real MobileFaceNet ONNX for production |
| System is not certified for production use | Research prototype only |

---

## Project Structure

```
Prajna 0.1/
├── app.py                          # Streamlit application (0.3)
├── frt_utils.py                    # 0.1 core logic (unchanged)
├── model_utils.py                  # 0.2 model/DB utilities (unchanged)
├── llm_utils.py                    # Multilingual explanations (unchanged)
├── requirements.txt                # All dependencies
├── MIGRATION_PLAN.md               # 0.2 → 0.3 migration documentation
│
├── core/
│   ├── stage1_model.py             # [NEW 0.3] MobileFaceNet wrapper
│   ├── database_manager.py         # [NEW 0.3] Dual-stage DB
│   ├── routing.py                  # [NEW 0.3] ρ score + routing decisions
│   ├── hierarchy.py                # [NEW 0.3] Pipeline orchestrator
│   ├── decision.py                 # 0.2 decision engine (unchanged)
│   ├── quality.py                  # 0.2 quality scoring (unchanged)
│   ├── thresholds.py               # 0.2 adaptive thresholds (unchanged)
│   ├── metrics.py                  # 0.2 metrics (unchanged)
│   └── calibration.py             # 0.2 calibration (unchanged)
│
├── utils/
│   ├── logging_utils_v3.py         # [NEW 0.3] Extended audit logger
│   ├── hierarchy_viz.py            # [NEW 0.3] Routing visualizations
│   ├── logging_utils.py            # 0.2 audit logger (unchanged)
│   ├── visualization.py            # 0.2 charts (unchanged)
│   └── language.py                 # Language utilities (unchanged)
│
├── evaluation/
│   ├── exp_hierarchy.py            # [NEW 0.3] 6 hierarchy experiments
│   └── experiments.py             # 0.2 experiments (unchanged)
│
├── tests/
│   ├── test_routing.py             # [NEW 0.3] 23 routing tests
│   └── test_database_manager.py    # [NEW 0.3] 9 DB tests
│
├── database/
│   ├── stage1/                     # [NEW 0.3] MobileFaceNet enrollments
│   └── stage2/                     # [NEW 0.3] IRESNET enrollments
│
├── models/
│   └── mobilefacenet.onnx          # [OPTIONAL] Download separately
│
└── logs/
    ├── audit_YYYYMMDD.jsonl        # 0.2 per-decision audit
    └── pipeline_YYYYMMDD.jsonl     # [NEW 0.3] Per-pipeline audit
```
