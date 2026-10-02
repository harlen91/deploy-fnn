import streamlit as st
import numpy as np
import joblib
import logging
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model

# Konfigurasi Halaman
st.set_page_config(page_title="Health Risk Predictor", layout="wide")

# ============================================================
# 1. Setup & Load Artifacts
# ============================================================
logging.basicConfig(filename='app_system.log', level=logging.INFO)

@st.cache_resource
def load_artifacts():
    model = load_model("model_fnn_risiko_kesehatan.keras")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artifacts()

# ============================================================
# 2. Sidebar UI (Input Control)
# ============================================================
st.sidebar.header("Input Data Pasien")
with st.sidebar.form("input_form"):
    umur = st.number_input("Umur", 1, 120, 30)
    tekanan_darah = st.number_input("Tekanan Darah (mmHg)", 50, 250, 120)
    kolesterol = st.number_input("Kolesterol (mg/dL)", 50, 400, 200)
    bmi = st.number_input("BMI", 10.0, 60.0, 22.5)
    gula_darah = st.number_input("Gula Darah (mg/dL)", 40, 400, 100)
    aktivitas_fisik = st.number_input("Aktivitas Fisik (jam/minggu)", 0, 168, 5)
    perokok = st.selectbox("Perokok?", ["Tidak", "Ya"])
    riwayat_keluarga = st.selectbox("Riwayat Penyakit Jantung?", ["Tidak", "Ya"])
    
    submit = st.form_submit_button("Analisis Risiko")

# ============================================================
# 3. Main Dashboard UI
# ============================================================
st.title("Dashboard Prediksi Risiko Kesehatan")
st.markdown("---")

if submit:
    # Persiapan Data
    p = 1 if perokok == "Ya" else 0
    rk = 1 if riwayat_keluarga == "Ya" else 0
    features = np.array([[umur, bmi, tekanan_darah, gula_darah, kolesterol, aktivitas_fisik, p, rk]])
    
    try:
        # Inferensi
        features_scaled = scaler.transform(features)
        prob = model.predict(features_scaled)
        idx = np.argmax(prob, axis=1)
        label = label_encoder.inverse_transform(idx)
        confidence = np.max(prob)
        
        # Layout Hasil
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Hasil Prediksi")
            st.metric(label="Tingkat Risiko", value=label[0])
            st.metric(label="Tingkat Kepercayaan", value=f"{confidence*100:.2f}%")
            
            if label[0] == "Tinggi":
                st.error("Saran: Segera konsultasikan dengan dokter spesialis.")
            elif label[0] == "Sedang":
                st.warning("Saran: Perbaiki pola makan dan tingkatkan aktivitas fisik.")
            else:
                st.success("Saran: Pertahankan gaya hidup sehat Anda.")

        with col2:
            st.subheader("Distribusi Probabilitas")
            df_prob = pd.DataFrame({'Risiko': label_encoder.classes_, 'Prob': prob[0]})
            fig = px.bar(df_prob, x='Risiko', y='Prob', color='Risiko', 
                         color_discrete_map={'Rendah': '#2ecc71', 'Sedang': '#f1c40f', 'Tinggi': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# ============================================================
# 4. Footer & Expander
# ============================================================
with st.expander("ℹ️ Tentang Model & Data"):
    st.write("Model ini menggunakan arsitektur Feed Forward Neural Network (FNN).")
    st.write("Data yang digunakan telah dinormalisasi menggunakan StandardScaler.")
    st.info("Catatan: Hasil prediksi ini adalah bantuan AI dan bukan pengganti diagnosis medis profesional.")
