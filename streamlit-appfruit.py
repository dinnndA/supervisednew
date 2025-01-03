import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'fruit.xlsx'
df = pd.read_excel(file_path)
X = df[['diameter', 'weight', 'red', 'green', 'blue']]  # Fitur
y = df['name']  # Label target

# Mapping label ke kelas secara manual
label_to_class = {'grapefruit': 0, 'orange': 1}
class_to_label = {v: k for k, v in label_to_class.items()}

# Fungsi untuk memuat model dan scaler
def load_model_and_scaler(model_file, scaler_file=None):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    scaler = None
    if scaler_file:
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler

# Fungsi untuk prediksi
def predict_fruit(features, model, scaler=None):
    # Pastikan fitur dalam format 2D
    features = [features] if isinstance(features[0], (int, float)) else features
    if scaler:
        features = scaler.transform(features)
    prediction_class = model.predict(features)[0]  # Prediksi kelas
    prediction_label = class_to_label[prediction_class]  # Mapping ke label
    return prediction_label, prediction_class

# Konfigurasi Streamlit
st.title("Aplikasi Prediksi Buah")
st.write("Pilih algoritma yang akan digunakan, lalu masukkan fitur buah untuk memprediksi jenis buah.")

# Pilih algoritma
algorithm = st.selectbox("Pilih Algoritma", ["Random Forest", "SVM", "Perceptron"])

# Load model dan scaler berdasarkan pilihan algoritma
if algorithm == "Random Forest":
    model_file = 'rffruit.pkl'
    model, scaler = load_model_and_scaler(model_file)
elif algorithm == "SVM":
    model_file = 'svmfruit.pkl'
    scaler_file = 'scaler_svm.pkl'
    model, scaler = load_model_and_scaler(model_file, scaler_file)
elif algorithm == "Perceptron":
    model_file = 'perceptronfruit.pkl'
    scaler_file = 'scaler_perceptron.pkl'
    model, scaler = load_model_and_scaler(model_file, scaler_file)

# Input pengguna
input_features = []
for col in X.columns:
    value = st.number_input(f"Masukkan nilai untuk {col}:", value=0.0)
    input_features.append(value)

# Prediksi
if st.button("Prediksi"):
    # Validasi input
    if any(v == 0.0 for v in input_features):
        st.warning("Pastikan semua fitur telah diisi dengan nilai yang sesuai.")
    else:
        label, class_index = predict_fruit(input_features, model, scaler)
        st.success(f"Model memprediksi jenis buah: {label} (Cluster: {class_index})")
