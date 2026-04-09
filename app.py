import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from model_utils import load_models, fetch_and_load_database, extract_face_and_embedding
from frt_utils import calculate_similarity_and_margin, compute_entropy_and_certainty, compute_quality, compute_dynamic_thresholds, responsibility_score, final_decision

st.set_page_config(page_title="प्रज्ञा 0.1 FRT", layout="wide")

# UI HEADER
st.markdown("""
<h1 style='font-size:48px;'>प्रज्ञा <span style="font-size:22px; color:gray;">FRT</span></h1>
<p style="font-size:20px;">Prajñā 0.1 — Responsibility-Aware Face Recognition</p>
<p style="color:#9ca3af; max-width:800px;">
A robust facial recognition pipeline with dynamic thresholding based on predictive entropy and image quality.
</p>
""", unsafe_allow_html=True)

with st.spinner("Initializing Models and Face Database..."):
    mtcnn, resnet = load_models()
    # load all identities, 1 sample each
    database_embs = fetch_and_load_database(num_classes=None, samples_per_class=1)

if not database_embs:
    st.error("Face database could not be loaded. Check Kaggle connection or dataset path.")
    st.stop()

st.sidebar.success(f"Database loaded with {len(database_embs)} identities.")

mode = st.radio("Select Mode", ["Live Face Verification", "Database Stats", "Full Dataset Analysis"], horizontal=True)

if mode == "Database Stats":
    st.subheader("Registered Identities")
    db_df = pd.DataFrame({"Identity": list(database_embs.keys())})
    st.dataframe(db_df, use_container_width=True)

elif mode == "Full Dataset Analysis":
    st.subheader("Run Responsibility Analysis on Directory")
    st.write("This mode actively queries every image within the `dataset/faces` directory against the database and computes the FRT metrics to measure accuracy strictly governed by the dynamic responsibility logic.")
    
    if st.button("Start Batch Analysis"):
        with st.spinner("Analyzing all faces in dataset..."):
            import os
            results = []
            target_folder = "dataset/faces"
            if not os.path.exists(target_folder):
                st.error("No dataset directory found.")
                st.stop()
                
            # Pre-calculate all images for accurate progress reporting
            all_images = []
            for person_name in os.listdir(target_folder):
                person_dir = os.path.join(target_folder, person_name)
                if not os.path.isdir(person_dir):
                    continue
                images = [f for f in os.listdir(person_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                for img_name in images:
                    all_images.append((person_name, img_name, os.path.join(person_dir, img_name)))
            
            total_images = len(all_images)
            if total_images == 0:
                st.warning("No images found in dataset/faces.")
                st.stop()
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (person_name, img_name, img_path) in enumerate(all_images):
                status_text.text(f"Analyzing {idx+1}/{total_images}: {person_name}/{img_name}")
                try:
                    img = Image.open(img_path).convert('RGB')
                    cropped_face, emb = extract_face_and_embedding(img, mtcnn, resnet)
                    if emb is not None:
                        scores, margin = calculate_similarity_and_margin(emb, database_embs)
                        top_k = min(5, len(scores))
                        top_scores_only = [s[1] for s in scores[:top_k]]
                        
                        H, U = compute_entropy_and_certainty(top_scores_only)
                        Q = compute_quality(cropped_face)
                        T_accept, T_review = compute_dynamic_thresholds(H, top_k, Q)
                        
                        best_match = scores[0][0]
                        # Clean up suffixes like _1 used for multi-sample logic
                        true_id_base = person_name.split("_")[0]
                        pred_id_base = best_match.split("_")[0]
                        
                        R = responsibility_score(scores[0][1], margin, U)
                        dec = final_decision(R, T_accept, T_review)
                        
                        is_correct = "Correct" if true_id_base == pred_id_base else "Incorrect"
                        
                        results.append({
                            "Image": img_name,
                            "True Identity": person_name,
                            "Predicted Identity": best_match,
                            "Is Correct": is_correct,
                            "Decision": dec,
                            "Score (R)": R
                        })
                except Exception:
                    pass
                
                # Update progress bar
                progress_bar.progress((idx + 1) / total_images)
                
            status_text.text(f"Completed analyzing {total_images} images.")
                        
            if not results:
                st.warning("No faces extracted.")
                st.stop()
                
            df = pd.DataFrame(results)
            st.success(f"Analysis complete on {len(df)} images!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Decision Breakdown")
                st.bar_chart(df["Decision"].value_counts())
            with col2:
                st.subheader("System Accuracy")
                st.bar_chart(df["Is Correct"].value_counts())
                
            st.subheader("Decision / Identity Validation")
            st.write("This table helps identify if wrong identities are being falsely **ACCEPTED** or if correct identities are cautiously **REVIEWED/REJECTED**.")
            st.dataframe(pd.crosstab(df["Is Correct"], df["Decision"]), use_container_width=True)
            
            st.subheader("Detailed Evaluation Results")
            st.dataframe(df.round(3), use_container_width=True)

elif mode == "Live Face Verification":
    st.subheader("Upload Face Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Resize original image to fit width better
            display_img = image.copy()
            display_img.thumbnail((400, 400))
            st.image(display_img, caption="Original Image", use_container_width=True)
            
        with st.spinner("Processing..."):
            cropped_face, emb = extract_face_and_embedding(image, mtcnn, resnet)
            
        with col2:
            if cropped_face is None or emb is None:
                st.error("No valid face detected in the image.")
            else:
                st.image(cropped_face, caption="Detected Face", width=150)
                
                # Metrics computation
                scores, margin = calculate_similarity_and_margin(emb, database_embs)
                top_k = min(5, len(scores))
                top_scores_only = [s[1] for s in scores[:top_k]]
                
                H, U = compute_entropy_and_certainty(top_scores_only)
                Q = compute_quality(cropped_face)
                
                top_match_name = scores[0][0]
                top_match_sim = scores[0][1]
                
                T_accept, T_review = compute_dynamic_thresholds(H, top_k, Q)
                R = responsibility_score(top_match_sim, margin, U)
                decision_label = final_decision(R, T_accept, T_review)
                
                # UI Rendering
                st.subheader("Identification Result")
                st.markdown(f"**Best Match:** `{top_match_name}`")
                
                st.markdown("### Top Matches")
                matches_df = pd.DataFrame(scores[:top_k], columns=["Identity", "Scaled Similarity"])
                st.dataframe(matches_df, use_container_width=True)
                
                st.markdown("### Responsibility Metrics")
                metrics_df = pd.DataFrame({
                    "Metric": ["Top-1 Similarity (S)", "Margin (M)", "Entropy (H)", "Certainty (U)", "Face Quality (Q)", "Responsibility Score (R)"],
                    "Value": [top_match_sim, margin, H, U, Q, R]
                }).round(4)
                st.dataframe(metrics_df, use_container_width=True)
                
                st.markdown("### Dynamic Thresholds")
                thresh_df = pd.DataFrame({
                    "Threshold": ["Accept Threshold", "Review Threshold"],
                    "Value": [T_accept, T_review]
                }).round(4)
                st.dataframe(thresh_df, use_container_width=True)
                
                st.markdown("### Final Decision")
                if decision_label == "ACCEPT":
                    st.success(f"ACCEPT — Identity verified proactively as {top_match_name}")
                elif decision_label == "REVIEW":
                    st.warning("REVIEW — Borderline confidence, human verification required")
                else:
                    st.error("REJECT — Low confidence or poor quality, identity unknown")
