import pandas as pd
from utils.preprocess import load_model, preprocess_input

def predict_from_features(df):
    model, scaler, feature_order = load_model()
    df_processed = preprocess_input(df, feature_order)
    X_scaled = scaler.transform(df_processed)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    df_result = df.copy()
    df_result['prediksi'] = preds
    df_result['prob_buka'] = probs[:, 0]
    df_result['prob_tutup'] = probs[:, 1]
    return df_result
