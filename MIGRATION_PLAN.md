# Prajñā Migration Plan: 0.2 → 0.3

## Unchanged Files (DO NOT MODIFY)

| File | Reason |
|---|---|
| `frt_utils.py` | Original 0.1 logic, preserved for backward compatibility |
| `model_utils.py` | Existing enrollment/extraction logic reused by database_manager.py |
| `llm_utils.py` | Multilingual explanation unchanged |
| `core/quality.py` | QualityComponents and compute_composite_quality reused as-is |
| `core/thresholds.py` | ThresholdRecord and compute_adaptive_thresholds reused as-is |
| `core/decision.py` | make_decision() called as final decision layer in hierarchy |
| `core/metrics.py` | All metric functions reused by hierarchy.py |
| `core/calibration.py` | Unchanged; exp_hierarchy.py imports from it |
| `utils/visualization.py` | Unchanged; new plots go in utils/hierarchy_viz.py |
| `utils/language.py` | Unchanged |
| `evaluation/experiments.py` | Unchanged; 0.3 adds exp_hierarchy.py alongside it |
| `requirements.txt` | Updated: add onnxruntime, pytest |

## New Files (CREATE)

| File | Phase | Purpose |
|---|---|---|
| `core/stage1_model.py` | 2 | MobileFaceNet wrapper (ONNX-based, lazy load) |
| `core/database_manager.py` | 3 | Dual-stage database: enroll, load, search |
| `core/routing.py` | 4 | ρ score, routing decision, dataclasses |
| `core/hierarchy.py` | 5 | hierarchical_inference() orchestrator |
| `utils/logging_utils_v3.py` | 6 | Extended audit: stage-level logging |
| `utils/hierarchy_viz.py` | 9 | Plotly helpers: Sankey, ρ histograms, ΔR plots |
| `evaluation/exp_hierarchy.py` | 7 | 6 experiments for 0.3 |
| `tests/test_routing.py` | 10 | pytest for routing module |
| `tests/test_hierarchy.py` | 10 | pytest for hierarchy orchestrator |
| `tests/test_database_manager.py` | 10 | pytest for dual DB |
| `tests/test_stage1.py` | 10 | pytest for Stage 1 model |
| `database/stage1/` | 3 | Stage-1 (MobileFaceNet) enrolled embeddings |
| `database/stage2/` | 3 | Stage-2 (InceptionResnetV1) enrolled embeddings |
| `models/` | 2 | MobileFaceNet ONNX weights directory |

## Modified Files

| File | Phase | Change |
|---|---|---|
| `app.py` | 8 | Add Tab 5 (🧭 Pipeline), extend Verify tab with routing cards, extend Analysis tab with hierarchy experiments |
| `requirements.txt` | 2 | Add: `onnxruntime`, `pytest`, `pytest-mock` |
| `README.md` | 11 | Full rewrite to describe 0.3 |

## Dependency Graph (0.3 additions only)

```
PIL Image
  → core/stage1_model.py     → MobileFaceNet ONNX embedding
  → model_utils.extract_face_full → InceptionResnetV1 embedding (unchanged)

core/quality.py              (unchanged, imported by hierarchy.py)
core/metrics.py              (unchanged, imported by hierarchy.py)
core/thresholds.py           (unchanged, imported by hierarchy.py)
core/decision.py             (unchanged, used as final stage)

core/routing.py              → ρ score, routing decision
  ← core/metrics.py          (R score formula)

core/database_manager.py
  ← model_utils.py           (extract_face_full, load_models)
  ← core/stage1_model.py     (Stage-1 embedding)
  ← core/quality.py          (is_enrollable check)

core/hierarchy.py            → orchestrates full pipeline
  ← core/stage1_model.py
  ← model_utils.py
  ← core/routing.py
  ← core/metrics.py
  ← core/quality.py
  ← core/thresholds.py
  ← core/decision.py
  ← core/database_manager.py

utils/logging_utils_v3.py    → extended JSONL audit
  ← core/hierarchy.py        (PipelineRecord)
  ← core/routing.py          (RoutingRecord)

evaluation/exp_hierarchy.py
  ← core/hierarchy.py
  ← core/database_manager.py
  ← utils/logging_utils_v3.py

app.py
  ← core/hierarchy.py        (primary inference call)
  ← core/database_manager.py (enrollment, DB load)
  ← utils/logging_utils_v3.py
  ← utils/hierarchy_viz.py
  ← evaluation/exp_hierarchy.py
```

## Risk Register

| Risk | Level | Mitigation |
|---|---|---|
| MobileFaceNet ONNX download fails | Medium | Graceful fallback: Stage 1 = InceptionResnetV1 at 112px |
| Dual DB embedding space mismatch | Low | Documented explicitly; ΔR treated as routing-space metric only |
| app.py modification breaks existing 0.2 tabs | Medium | New hierarchy call is additive; existing `combined_db` path preserved |
| Stage-1 DB empty on first run | Low | Warn in UI; hierarchy degrades gracefully to Stage 2 only |
| pytest fails on Streamlit imports | Low | Model loading mocked in all tests |
