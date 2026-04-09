import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("vasukipatel/face-recognition-dataset")
print("Downloaded path:", path)
for root, dirs, files in os.walk(path):
    print(f"Directory: {root}, Files count: {len(files)}")
    if files:
        print(f"Sample files: {files[:3]}")

try:
    # We will try loading it if there is a CSV, but likely it's just image folders
    print("Finding CSV files...")
    csvs = [f for f in os.listdir(path) if f.endswith('.csv')]
    print("CSV files:", csvs)
except Exception as e:
    print(e)
