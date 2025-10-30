import streamlit as st
import pandas as pd
import numpy as np
import librosa

from utils.predict import predict_from_features

st.set_page_config(page_title="Voice Identification: Buka/Tutup", layout="wide")
st.title("🔊 Voice Identification — Deteksi Suara Buka/Tutup")

st.write("Upload file **audio (.wav)** untuk mendeteksi apakah suara termasuk kategori *Buka* atau *Tutup*.")

uploaded = st.file_uploader("Upload file WAV", type=["wav"])

def extract_features(y, sr):
    """Ekstraksi fitur dasar dari audio"""
    features = {
        'zcr_mean': np.mean(librosa.feature.zero_crossing_rate(y)),
        'zcr_std': np.std(librosa.feature.zero_crossing_rate(y)),
        'rmse_mean': np.mean(librosa.feature.rms(y=y)),
        'rmse_std': np.std(librosa.feature.rms(y=y)),
        'centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'centroid_std': np.std(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'bandwidth_mean': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'bandwidth_std': np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'rolloff_std': np.std(librosa.feature.spectral_rolloff(y=y, sr=sr)),
    }

    # Tambahkan beberapa koefisien MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(1, 14):
        features[f'mfcc{i}_mean'] = np.mean(mfcc[i-1])
        features[f'mfcc{i}_std'] = np.std(mfcc[i-1])
    
    return pd.DataFrame([features])

if uploaded:
    st.audio(uploaded, format="audio/wav")

    with st.spinner("🔍 Mengekstraksi fitur dari audio..."):
        y, sr = librosa.load(uploaded, sr=None)
        df_features = extract_features(y, sr)
    
    st.subheader("📊 Fitur yang Diekstraksi")
    st.dataframe(df_features)

    with st.spinner("🤖 Melakukan prediksi..."):
        result = predict_from_features(df_features)

    st.subheader("🔎 Hasil Prediksi")
    st.dataframe(result)

    pred_class = result['prediksi'].iloc[0]
    st.success(f"Hasil prediksi: **{pred_class}**")
else:
    st.info("Silakan upload file audio (.wav) untuk mendeteksi apakah suara adalah *Buka* atau *Tutup*.")
