import numpy as np
import cv2

def calculate_similarity_and_margin(query_emb, database_embs):
    """
    Computes scaled cosine similarity [0, 1] and top-2 margin.
    query_emb: numpy array (1, 512)
    database_embs: dict of {name: numpy array (1, 512)}
    Returns: list of tuples (name, normalized_similarity) sorted desc, and Margin
    """
    scores = []
    query_vec = query_emb.flatten()
    
    for name, emb in database_embs.items():
        db_vec = emb.flatten()
        # Cosine similarity
        cos_sim = np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec))
        # Scale range [-1, 1] to [0, 1]
        scaled_sim = (cos_sim + 1) / 2
        scores.append((name, scaled_sim))
        
    # Sort descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Margin
    margin = 0.0
    if len(scores) > 1:
        margin = scores[0][1] - scores[1][1]
    
    return scores, float(margin)

def compute_entropy_and_certainty(similarities, tau=0.03):
    """
    Computes Shannon Entropy and normalized Certainty from top-K similarities.
    similarities: list of [0, 1] scaled similarities for top K matches.
    tau: Softmax temperature stringency
    """
    if len(similarities) == 0:
        return 1.0, 0.0
    if len(similarities) == 1:
        return 0.0, 1.0
        
    sims = np.array(similarities)
    # Apply softmax with temperature
    exp_sims = np.exp(sims / tau)
    probs = exp_sims / np.sum(exp_sims)
    
    # Entropy
    eps = 1e-9
    probs = np.clip(probs, eps, 1)
    H = -np.sum(probs * np.log(probs))
    
    # Certainty
    U = 1 - H / np.log(len(probs))
    return float(H), float(U)

def compute_quality(image_array):
    """
    Computes normalized quality score [0, 1] based on Laplacian variance (blur).
    image_array: numpy array (H, W, 3) RGB
    """
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: empirically, variance > 1000 is sharp.
    max_expected_variance = 100.0
    quality = min(variance / max_expected_variance, 1.0)
    return float(quality)

def compute_dynamic_thresholds(H, K, Q_norm):
    """
    Computes dynamic T_accept and T_review based on Entropy and Quality.
    H: Entropy
    K: Num classes considered for entropy 
    Q_norm: Normalized image quality [0, 1]
    """
    T_base_acc = 0.60
    T_base_rev = 0.35
    alpha = 0.1
    beta = 0.15
    
    # Normalized entropy
    H_norm = H / np.log(K) if K > 1 else 0
    
    penalty = alpha * H_norm + beta * (1 - Q_norm)
    
    T_accept = min(T_base_acc + penalty, 1.0)
    T_review = min(T_base_rev + penalty, 1.0)
    
    # ensure review is strictly lower than or equal to accept
    T_review = min(T_review, T_accept)
    
    return float(T_accept), float(T_review)

def responsibility_score(scaled_top1_sim, margin, certainty):
    """
    Computes final responsibility score dynamically.
    """
    return 0.6 * scaled_top1_sim + 0.2 * margin + 0.2 * certainty

def final_decision(R, T_accept, T_review):
    """
    Determines ACCEPT / REVIEW / REJECT class based on dynamic thresholding.
    """
    if R >= T_accept:
        return "ACCEPT"
    elif R >= T_review:
        return "REVIEW"
    else:
        return "REJECT"
