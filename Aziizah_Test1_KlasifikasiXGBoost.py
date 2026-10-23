import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Mengatur tampilan halaman Streamlit
st.set_page_config(page_title="Dashboard Klasifikasi XGBoost", layout="wide")

# ==========================================
# 1. Judul dan Deskripsi Proyek
# ==========================================
st.title("🌧️ Dashboard Klasifikasi Hujan menggunakan XGBoost")
st.write("Proyek ini menggunakan algoritma XGBoost untuk memprediksi apakah akan turun hujan atau tidak berdasarkan data cuaca harian di Kolkata (Jan - Jun 2026).")
st.markdown("---")

# ==========================================
# Load & Preprocess Data
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("kolkata_weather_jan_june_2026.csv")
    df['is_rain'] = (df['precipitation_sum'] > 0).astype(int)
    return df

df = load_data()
fitur = ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 
         'sunshine_duration', 'wind_speed_10m_max', 'wind_direction_10m_dominant']

# ==========================================
# 2. Ringkasan Dataset
# ==========================================
st.header("📊 Ringkasan Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Data (Baris)", df.shape[0])
col2.metric("Jumlah Fitur Prediksi", len(fitur))
col3.metric("Proporsi Kelas (Hujan : Tidak Hujan)", f"{df['is_rain'].sum()} : {df.shape[0] - df['is_rain'].sum()}")

with st.expander("Klik untuk melihat cuplikan tabel data"):
    st.dataframe(df[fitur + ['is_rain']].head(10))

# ==========================================
# 3. Grafik EDA
# ==========================================
st.header("📈 Eksplorasi Data (EDA)")
tab1, tab2, tab3 = st.tabs(["Proporsi Kelas Target", "Distribusi Fitur", "Korelasi Fitur"])

with tab1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='is_rain', ax=ax, palette="Set2")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Tidak Hujan (0)', 'Hujan (1)'])
    st.pyplot(fig)

with tab2:
    st.write("Distribusi fitur numerik yang digunakan untuk pelatihan model:")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(fitur):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribusi {col}')
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[fitur + ['is_rain']].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ==========================================
# Pemodelan XGBoost
# ==========================================
X = df[fitur]
y = df['is_rain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# 4, 5, 6. Evaluasi & Visualisasi Hasil Model
# ==========================================
st.header("⚙️ Performa Model & Visualisasi Prediksi")

col_metric, col_cm = st.columns([1, 2])
with col_metric:
    st.subheader("Nilai Performa Model")
    metrics_df = pd.DataFrame({
        "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC Score"],
        "Nilai": [f"{accuracy_score(y_test, y_pred):.4f}", 
                  f"{precision_score(y_test, y_pred, zero_division=0):.4f}", 
                  f"{recall_score(y_test, y_pred, zero_division=0):.4f}", 
                  f"{f1_score(y_test, y_pred, zero_division=0):.4f}", 
                  f"{roc_auc_score(y_test, y_prob):.4f}"]
    })
    st.table(metrics_df)

with col_cm:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Prediksi Model')
    ax.set_ylabel('Aktual Data')
    st.pyplot(fig)

col_roc, col_feat = st.columns(2)
with col_roc:
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='darkorange', label=f'ROC AUC = {roc_auc_score(y_test, y_prob):.2f}')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    st.pyplot(fig)

with col_feat:
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(5, 4))
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    sns.barplot(x=importances[sorted_idx], y=np.array(fitur)[sorted_idx], ax=ax, palette="viridis")
    st.pyplot(fig)

st.markdown("---")

# ==========================================
# 7. Form Input Manual Prediksi
# ==========================================
st.header("🔮 Coba Prediksi Manual")
st.write("Masukkan nilai fitur di bawah ini untuk melihat apakah model memprediksi hari tersebut hujan atau tidak.")

with st.form("prediction_form"):
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        in_temp_mean = st.number_input("Suhu Rata-rata (temperature_2m_mean)", value=float(df['temperature_2m_mean'].mean()))
        in_temp_max = st.number_input("Suhu Maksimum (temperature_2m_max)", value=float(df['temperature_2m_max'].max()))
        in_temp_min = st.number_input("Suhu Minimum (temperature_2m_min)", value=float(df['temperature_2m_min'].min()))
        
    with col_input2:
        in_sunshine = st.number_input("Durasi Sinar Matahari (sunshine_duration)", value=float(df['sunshine_duration'].mean()))
        in_wind_speed = st.number_input("Kecepatan Angin Maks (wind_speed_10m_max)", value=float(df['wind_speed_10m_max'].mean()))
        in_wind_dir = st.number_input("Arah Angin Dominan (wind_direction_10m_dominant)", value=int(df['wind_direction_10m_dominant'].mean()))
    
    submit_button = st.form_submit_button(label="Prediksi Cuaca")

if submit_button:
    # Memasukkan input ke dalam dataframe untuk diprediksi
    input_data = pd.DataFrame({
        'temperature_2m_mean': [in_temp_mean],
        'temperature_2m_max': [in_temp_max],
        'temperature_2m_min': [in_temp_min],
        'sunshine_duration': [in_sunshine],
        'wind_speed_10m_max': [in_wind_speed],
        'wind_direction_10m_dominant': [in_wind_dir]
    })
    
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100
    
    if pred == 1:
        st.error(f"⚠️ **Model memprediksi HUJAN!** (Peluang hujan: {prob:.2f}%)")
    else:
        st.success(f"☀️ **Model memprediksi TIDAK HUJAN.** (Peluang hujan: {prob:.2f}%)")