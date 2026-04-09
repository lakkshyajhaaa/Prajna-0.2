# Prajñā 0.1 — Responsibility-Aware Face Recognition System

Prajñā 0.1 is a robust facial recognition pipeline built around the philosophy of **Responsibility-Aware Thresholding**. Rather than simply returning a top match based on standard cosine similarity, Prajñā actively guards against false positives by evaluating predictive entropy (uncertainty), margin between competing identities, and input image quality.

## Core Architecture
- **Face Extraction:** MTCNN (Multi-task Cascaded Convolutional Networks) handles face detection and alignment.
- **Embedding Generation:** InceptionResnetV1 (pretrained on VGGFace2) generates 512-dimensional feature vectors.
- **Decision Engine:** Custom `frt_utils.py` logic combining similarity, margin, entropy, and Laplacian variance (blur) into a unified **Responsibility Score (R)**.
- **Tri-State Outputs:** Instead of a binary match/no-match, the system outputs:
  - **ACCEPT:** High confidence match, proactive system trust.
  - **REVIEW:** Borderline match, flagged for human verification.
  - **REJECT:** Unrecognized or mathematically invalid prediction.

---

## The Responsibility Score (R) & Dynamic Heuristics

The heart of the application is the dynamic calculation. To maximize true positive approvals (Accepts) while actively defending against false positives (strangers passing the threshold), we implemented highly tailored heuristic values.

### 1. Responsibility Weighting (0.70 / 0.15 / 0.15)
The final responsibility score (`R`) is computed using a weighted formula:
`R = (0.70 * Similarity) + (0.15 * Margin) + (0.15 * Certainty)`

**Why these values?**
Previously, similarity, margin, and certainty were weighted 0.6 / 0.2 / 0.2. However, single-sample datasets (where there is only one photo per identity) caused the `Margin` metric to wildly fluctuate. We recalibrated the scale to **70% Similarity**, strongly prioritizing the raw mathematical vector. The remaining 30% acts as a dual-guard for isolated certainty and margin validation, preventing extreme penalties if a dataset has similar-looking faces.

### 2. The Acceptance Floor (`T_acc = 0.72`)
Cosine distance natively outputs values between `[-1, 1]`. To create normalized scores, the system scales similarity to `[0, 1]`. This mathematical shift causes complete strangers (scoring 0.4 un-scaled) to artificially jump up to **~0.70 scaled similarity**. 

**Why 0.72 and 0.62?**
By setting the **Accept Threshold (`T_base_acc`) to `0.72`** and the **Review Threshold (`T_base_rev`) to `0.62`**, we prevent "strangers" (who typically score ~0.67 in this scaled system) from ever passing the Accept barrier. True matches typically score `0.75 - 0.85+`, allowing them to cleanly glide over the 0.72 floor.

### 3. Maximum Expected Variance (`50.0`)
A blur penalty is applied to images that drop below a specific Laplacian variance score. 

**Why 50.0?**
Originally set to `100.0`, standard webcam snaps and moderately sized dataset photos were persistently failing the quality check. This caused an excessive "blur penalty" that spiked the Accept threshold upwards artificialy. By locking the max expected variance to `50.0`, typical digital quality is accepted as "perfect", reserving penalties exclusively for actively motion-blurred or degraded assets.

### 4. Softmax Temperature (`tau = 0.03`)
Certainty (Shannon Entropy) is measured by applying softmax across the top identities.

**Why 0.03?**
Using a standard temperature like `0.1` spread probabilities across matches too smoothly. Even if a clear winner existed, the system generated "phantom entropy" (artificial uncertainty) which dragged the Responsibility score down. Dropping `tau` to `0.03` sharpens the calculation, causing Certainty metrics to accurately peak to `>0.95` when a single match is legitimately identified.

---

## Developer Guide

### Running the System
You can interface with Prajñā via a Streamlit Dashboard.
```bash
streamlit run app.py
```

### Dashboard Modes:
1. **Live Face Verification:** Upload individual images for breakdown of Margin, Entropy, Quality, and Responsibility scoring.
2. **Database Stats:** View actively enrolled dataset profiles.
3. **Full Dataset Analysis:** Runs a massive bulk verification loop analyzing all faces in the active dataset, calculating aggregate True Positive Accepts and identifying any Reject/Review behaviors across the dataset.
