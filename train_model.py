import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# === 1. Load dataset ===
data_path = os.path.join('data', 'bukatutup.csv')
df = pd.read_csv(data_path)

# === 2. Pisahkan fitur dan label ===
X = df.drop(columns=['file_name', 'class'])
y = df['class']

# === 3. Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === 4. Standarisasi + model ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_scaled, y_train)

# === 5. Evaluasi ===
acc = clf.score(X_test_scaled, y_test)
print(f"Akurasi model: {acc:.2f}")

# === 6. Simpan model ===
os.makedirs('models', exist_ok=True)
joblib.dump({'model': clf, 'scaler': scaler, 'features': X.columns.tolist()}, 'models/classifier.pkl')
print("✅ Model disimpan di models/classifier.pkl")
