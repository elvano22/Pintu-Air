import streamlit as st
import pandas as pd
from utils.footer import show_footer

st.title("📖 Panduan Penggunaan Website")
st.markdown("---")

st.markdown("""
Website **Pintu Air Manggarai** merupakan sistem monitoring dan prediksi tinggi muka air 
yang dirancang untuk membantu pemantauan kondisi Sungai Ciliwung secara terpadu, 
khususnya di wilayah Pintu Air Manggarai, Jakarta.
Sistem ini menggabungkan data historis, kondisi cuaca, dan model machine learning 
untuk mendukung kewaspadaan dini terhadap potensi banjir.
""")

# === OVERVIEW SISTEM ===
st.subheader("🌊 Gambaran Umum Sistem")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ### **Lokasi Stasiun Pemantauan**
    - **Katulampa** (hulu sungai)
    - **Depok** (tengah aliran)
    - **Manggarai** (hilir – Jakarta Pusat)
    
    ### **Jenis Data yang Ditampilkan**
    - Tinggi muka air (cm)
    - Kondisi cuaca
    - Status siaga (Normal, Siaga 3, Siaga 2, Siaga 1)
    - Prediksi tinggi muka air
    """)

with col2:
    st.markdown("""
    ### **Teknologi Sistem**
    - Model **XGBoost** untuk prediksi
    - Pemantauan data secara berkala (per jam)
    - Visualisasi interaktif menggunakan Plotly
    - Prediksi multi-horizon (6–72 jam ke depan)
    
    ### **Pengguna Sasaran**
    - Petugas hidrologi dan kebencanaan
    - Masyarakat umum
    - Peneliti dan akademisi
    """)

st.markdown("---")

# === PAGE 1: BERANDA ===
st.subheader("🏠 Halaman 1: Beranda (Dashboard Utama)")

st.markdown("""
Beranda merupakan **dashboard utama** yang menampilkan kondisi terkini 
serta prediksi tinggi muka air di seluruh stasiun pemantauan secara ringkas dan informatif.
""")

with st.expander("📊 Fitur Halaman Beranda", expanded=True):
    st.markdown("""
    #### **1. Prediksi Tinggi Muka Air Manggarai**
    - Menggunakan model XGBoost yang telah dioptimasi
    - Menampilkan hasil prediksi dengan performa terbaik
    
    #### **2. Prediksi Jangka Pendek (6 Jam)**
    - Tabel prediksi per jam hingga 6 jam ke depan
    - Grafik interaktif yang menampilkan data historis dan hasil prediksi
    - Garis batas status siaga Manggarai (750, 850, 950 cm)
    
    #### **3. Sistem Peringatan Dini**
    - Peringatan otomatis ketika prediksi melewati batas siaga
    - Informasi estimasi waktu terjadinya status siaga
    
    #### **4. Informasi Model dan Evaluasi**
    - Visualisasi data pelatihan, pengujian, dan prediksi
    - Interval kepercayaan hasil prediksi
    - Metrik evaluasi model (RMSE dan R²)
    
    #### **5. Peta Interaktif Stasiun**
    - Lokasi seluruh stasiun pemantauan
    - Warna penanda sesuai status siaga
    - Informasi tinggi muka air dan cuaca terkini
    
    #### **6. Status Terkini Seluruh Stasiun**
    - Kondisi Katulampa, Depok, dan Manggarai
    - Perubahan tinggi muka air dibandingkan jam sebelumnya
    - Informasi cuaca pada masing-masing lokasi
    """)

st.markdown("### Cara Menggunakan Halaman Beranda")

tab1, tab2, tab3 = st.tabs(["🔍 Pengecekan Cepat", "📈 Analisis Prediksi", "🗺️ Peta Monitoring"])

with tab1:
    st.markdown("""
    **Langkah pengecekan singkat:**
    1. Periksa tabel dan grafik prediksi 6 jam ke depan
    2. Perhatikan notifikasi peringatan dini jika tersedia
    3. Lihat status terbaru pada bagian ringkasan stasiun
    """)

with tab2:
    st.markdown("""
    **Untuk analisis lebih mendalam:**
    1. Buka informasi model prediksi
    2. Amati grafik performa model
    3. Perhatikan interval kepercayaan hasil prediksi
    """)

with tab3:
    st.markdown("""
    **Untuk pemantauan berbasis peta:**
    1. Arahkan kursor ke titik stasiun untuk melihat detail
    2. Gunakan warna sebagai indikator status siaga
    3. Manfaatkan fitur zoom untuk melihat area tertentu
    """)

st.markdown("---")

# === PAGE 2: ANALISIS DATA ===
st.subheader("📊 Halaman 2: Informasi dan Analisis Data")

st.markdown("""
Halaman ini digunakan untuk **eksplorasi dan analisis data** deret waktu, 
guna memahami karakteristik data, pola hubungan antar stasiun, 
serta perilaku musiman sebelum dilakukan pemodelan.
""")

with st.expander("🔍 Fitur Halaman Analisis Data", expanded=True):
    st.markdown("""
    #### **1. Sumber Data**
    - Menggunakan dataset bawaan sistem atau mengunggah data sendiri
    - Sistem memeriksa kesesuaian format data
    
    #### **2. Pemeriksaan Data Hilang**
    - Deteksi otomatis nilai yang hilang
    - Ringkasan jumlah dan distribusi data kosong
    
    #### **3. Informasi Umum Dataset**
    - Jumlah baris dan kolom
    - Rentang waktu data
    
    #### **4. Analisis Korelasi**
    - Heatmap korelasi antar stasiun
    - Nilai korelasi ditampilkan secara interaktif
    
    #### **5. Visualisasi Deret Waktu**
    - Grafik interaktif seluruh stasiun
    - Fitur zoom dan perbandingan antar lokasi
    
    #### **6. Analisis Detail per Stasiun**
    - Pola deret waktu
    - Distribusi data dan deteksi outlier
    - Tren bulanan
    
    #### **7. Analisis Cross-Correlation**
    - Analisis hubungan antar stasiun dengan pergeseran waktu (lag)
    - Identifikasi lag dengan korelasi tertinggi
    
    #### **8. Dekomposisi Deret Waktu**
    - Pemisahan komponen tren, musiman, dan residual
    - Pilihan periode harian atau mingguan
    
    #### **9. Statistik Deskriptif**
    - Ringkasan statistik dasar setiap stasiun
    - Perbandingan karakteristik data
    """)

st.markdown("### Cara Menggunakan Halaman Analisis")

tab1, tab2, tab3 = st.tabs(["🚀 Mulai Analisis", "🔎 Eksplorasi Data", "🧪 Analisis Lanjutan"])

with tab1:
    st.markdown("""
    1. Pilih sumber data (unggah atau gunakan data bawaan)
    2. Periksa ringkasan dan kelengkapan data
    """)

with tab2:
    st.markdown("""
    1. Amati korelasi antar stasiun
    2. Tinjau grafik deret waktu
    3. Lakukan analisis detail pada stasiun tertentu
    """)

with tab3:
    st.markdown("""
    1. Gunakan cross-correlation untuk menentukan lag optimal
    2. Lakukan dekomposisi untuk memahami pola tren dan musiman
    3. Gunakan statistik deskriptif sebagai dasar pemodelan
    """)

st.markdown("---")

show_footer()
