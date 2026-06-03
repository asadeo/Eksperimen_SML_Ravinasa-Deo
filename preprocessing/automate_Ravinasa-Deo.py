import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def fetch_data():
    url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
    file_name = "students_dropout.zip"
    print("Mengunduh dataset dari UCI ML Repository...")
    urllib.request.urlretrieve(url, file_name)
    
    print("Mengekstrak dataset...")
    os.makedirs("dataset_raw", exist_ok=True)
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall("dataset_raw")
    return "dataset_raw/data.csv"

def preprocess_data(csv_path):
    print("Memulai proses pembersihan data...")
    df = pd.read_csv(csv_path, sep=';')
    df_clean = df.copy()
    
    # Filter Target
    df_clean = df_clean[df_clean['Target'] != 'Enrolled']
    
    # Label Encoding
    le = LabelEncoder()
    df_clean['Target'] = le.fit_transform(df_clean['Target'])
    
    # Pemisahan Fitur dan Target
    X = df_clean.drop('Target', axis=1)
    y = df_clean['Target']
    
    # Standardisasi Skala
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Penggabungan Kembali
    df_final = X_scaled.copy()
    df_final['Target'] = y.values
    
    return df_final

def save_data(df, output_dir="preprocessing"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data_siap_latih.csv")
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Data bersih berhasil disimpan di {output_path}")

if __name__ == "__main__":
    raw_csv = fetch_data()
    clean_df = preprocess_data(raw_csv)
    save_data(clean_df)