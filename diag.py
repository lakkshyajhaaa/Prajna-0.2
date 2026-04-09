import sys
import os
import torch
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())
from model_utils import load_models, extract_face_and_embedding
from frt_utils import calculate_similarity_and_margin, compute_entropy_and_certainty, compute_quality, compute_dynamic_thresholds, responsibility_score

mtcnn, resnet = load_models()
database_dir = "dataset/faces"

def run_test():
    db_embs = {}
    persons = os.listdir(database_dir)[:3]
    for p in persons:
        pdir = os.path.join(database_dir, p)
        if not os.path.isdir(pdir): continue
        imgs = [f for f in os.listdir(pdir) if f.endswith('.jpg')]
        if not imgs: continue
        # Load one image into database
        img = Image.open(os.path.join(pdir, imgs[0])).convert('RGB')
        _, emb = extract_face_and_embedding(img, mtcnn, resnet)
        if emb is not None:
            db_embs[p] = emb
            print(f"Loaded {p} to DB")
            
    # Now query with the second image of the first person
    qperson = persons[0]
    qimgs = [f for f in os.listdir(os.path.join(database_dir, qperson)) if f.endswith('.jpg')]
    if len(qimgs) > 1:
        qimg_path = os.path.join(database_dir, qperson, qimgs[1])
        print(f"\nQuerying with {qperson} - {qimgs[1]}")
        img = Image.open(qimg_path).convert('RGB')
        cface, emb = extract_face_and_embedding(img, mtcnn, resnet)
        
        scores, margin = calculate_similarity_and_margin(emb, db_embs)
        print("Scores:")
        for s in scores: print("  ", s)
        print("Margin:", margin)
        
        top_k = min(5, len(scores))
        top_scores_only = [s[1] for s in scores[:top_k]]
        
        H, U = compute_entropy_and_certainty(top_scores_only, tau=0.1)
        Q = compute_quality(cface)
        
        R = responsibility_score(scores[0][1], margin, U)
        T_a, T_r = compute_dynamic_thresholds(H, top_k, Q)
        print(f"H={H:.2f}, U={U:.2f}, Q={Q:.2f}")
        print(f"R = {R:.3f} | T_acc = {T_a:.3f} | T_rev = {T_r:.3f}")
        if R >= T_a:
            print("Decision: ACCEPT")
        elif R >= T_r:
            print("Decision: REVIEW")
        else:
            print("Decision: REJECT")
    else:
        print("Not enough images for person", qperson)

run_test()
