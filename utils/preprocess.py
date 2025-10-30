import pandas as pd
import joblib
import os

def load_model():
    """Memuat model dan scaler"""
    model_path = os.path.join('models', 'classifier.pkl')
    data = joblib.load(model_path)
    return data['model'], data['scaler'], data['features']

def preprocess_input(df, feature_order):
    """Pastikan urutan kolom input sama seperti saat training"""
    df = df[feature_order]
    return df
