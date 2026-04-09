import torch
import numpy as np
import kagglehub
import os
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import streamlit as st

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    mtcnn_model = MTCNN(keep_all=False, device=device)
    resnet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn_model, resnet_model

@st.cache_data
def fetch_and_load_database(num_classes=None, samples_per_class=1):
    """
    Downloads kaggle face dataset to `./dataset/faces` locally and builds an in-memory embedding database.
    Returns: dict of {name: embedding_array (1, 512)}
    """
    local_dataset_path = "dataset/faces"
    
    if not os.path.exists(local_dataset_path) or len(os.listdir(local_dataset_path)) == 0:
        try:
            dataset_path = kagglehub.dataset_download("vasukipatel/face-recognition-dataset")
            # Safely find the image directory 
            target_folder = dataset_path
            for root, dirs, files in os.walk(dataset_path):
                if "Original Images" in dirs:
                    target_folder = os.path.join(root, "Original Images")
                    if "Original Images" in os.listdir(target_folder):
                        target_folder = os.path.join(target_folder, "Original Images")
                    break
            
            import shutil
            shutil.copytree(target_folder, local_dataset_path, dirs_exist_ok=True)
        except Exception as e:
            st.error(f"Failed to download dataset: {e}")
            return {}
            
    target_folder = local_dataset_path
            
    database = {}
    classes_loaded = 0
    mtcnn, resnet = load_models()
    
    if not os.path.exists(target_folder):
        return database
        
    for person_name in os.listdir(target_folder):
        person_dir = os.path.join(target_folder, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        if not images:
            continue
            
        success_count = 0
        for img_name in images:
            img_path = os.path.join(person_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Get embeddings using facenet-pytorch
                face_tensor = mtcnn(img)
                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = resnet(face_tensor).cpu().numpy()
                    
                    if samples_per_class > 1:
                        database[f"{person_name}_{success_count+1}"] = emb
                    else:
                        database[person_name] = emb
                        
                    success_count += 1
                    if success_count >= samples_per_class:
                        break
            except Exception:
                pass
                
        if success_count > 0:
            classes_loaded += 1
            if num_classes is not None and classes_loaded >= num_classes:
                break
                
    return database

def extract_face_and_embedding(image, mtcnn_model, resnet_model):
    """
    Given a PIL image, extracts face array and computes embedding.
    Returns (cropped_face_np_array, embedding_np_array) or (None, None)
    """
    boxes, probs = mtcnn_model.detect(image)
    if boxes is None or len(boxes) == 0:
        return None, None
        
    box = boxes[0]
    box = [int(b) for b in box]
    
    img_arr = np.array(image)
    h, w, _ = img_arr.shape
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    cropped_face = img_arr[y1:y2, x1:x2]
    
    face_tensor = mtcnn_model(image)
    if face_tensor is None:
        return np.array(cropped_face), None
        
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet_model(face_tensor).cpu().numpy()
        
    return np.array(cropped_face), emb
