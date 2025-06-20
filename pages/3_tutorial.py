import streamlit as st
import pandas as pd
from utils.footer import show_footer

st.title("📖 Panduan Penggunaan Website")
st.markdown("---")

st.markdown("""
Website **Pintu Air Manggarai** adalah sistem monitoring dan prediksi tinggi muka air yang dirancang untuk 
membantu pemantauan kondisi aliran Sungai Ciliwung, khususnya di area Pintu Air Manggarai, Jakarta.
""")

# === OVERVIEW SISTEM ===
st.subheader("🌊 Overview Sistem")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ### **Lokasi Monitoring**
    - **Katulampa** (Hulu Sungai)
    - **Depok** (Tengah Aliran) 
    - **Manggarai** (Hilir - Jakarta Pusat)
    
    ### **Data Yang Dipantau**
    - Tinggi muka air (cm)
    - Kondisi cuaca
    - Status siaga (Normal, Siaga 3, 2, 1)
    - Prediksi masa depan
    """)

with col2:
    st.markdown("""
    ### **Teknologi Yang Digunakan**
    - **XGBoost Machine Learning** untuk prediksi
    - **Real-time monitoring** dengan update per jam
    - **Interactive visualization** dengan Plotly
    - **Multi-horizon forecasting** (6-72 jam)
    
    ### **Target Pengguna**
    - Petugas BMKG/Hidrologi
    - Masyarakat Jakarta
    - Peneliti dan akademisi
    """)

st.markdown("---")

# === PAGE 1: BERANDA ===
st.subheader("🏠 Halaman 1: Beranda (Dashboard Utama)")

st.markdown("### **Fungsi Utama:**")
st.markdown("""
Beranda adalah **dashboard utama** yang memberikan gambaran komprehensif tentang kondisi terkini dan prediksi 
tinggi muka air di seluruh stasiun monitoring.
""")

with st.expander("📊 Fitur-Fitur Halaman Beranda", expanded=True):
    st.markdown("""
    #### **1. 📈 Prediksi Tinggi Muka Air Manggarai**
    - **Model XGBoost** yang telah dioptimasi untuk akurasi tinggi
    - **Performa superior** dibanding LSTM dan SARIMAX
    - Parameter model yang telah di-tuning melalui grid search
    
    #### **2. 🎯 Prediksi Jangka Pendek (6 Jam)**
    - **Tabel detail** prediksi per jam (1, 2, 3, 4, 5, 6 jam ke depan)
    - **Range confidence interval** untuk setiap prediksi
    - **Plot interaktif** menampilkan 24 jam terakhir + 6 jam forecast
    - **Garis batas siaga** (750, 850, 950 cm) untuk Manggarai
    
    #### **3. ⚠️ Sistem Peringatan Dini**
    - **Alert otomatis** jika prediksi mencapai batas siaga
    - **Notifikasi waktu spesifik** (contoh: "Siaga 2 dalam 3 jam")
    - **Status real-time** kondisi normal/siaga
    
    #### **4. 📊 Analisis Prediksi Komprehensif**
    - **Visualisasi full timeline**: Training → Testing → Forecasting
    - **Confidence intervals** 95% untuk testing dan forecasting
    - **Performance metrics**: RMSE, R² Score
    - **Model parameters** dan informasi teknis
    
    #### **5. 🗺️ Peta Interaktif Stasiun**
    - **Lokasi geografis** semua stasiun monitoring
    - **Color-coded status** berdasarkan level siaga
    - **Hover information** dengan data terkini
    - **Weather information** per stasiun
    
    #### **6. 📋 Status Real-time Semua Stasiun**
    - **Katulampa**: Status hulu sungai dengan threshold khusus
    - **Depok**: Status tengah aliran dengan threshold khusus  
    - **Manggarai**: **Focus utama** dengan emphasis khusus
    - **Delta changes** dari jam sebelumnya
    - **Weather conditions** terkini
    - **Expandable thresholds** untuk setiap stasiun
    """)

st.markdown("### **Cara Menggunakan Halaman Beranda:**")

tab1, tab2, tab3 = st.tabs(["🎯 Quick Check", "📈 Analisis Prediksi", "🗺️ Monitoring Map"])

with tab1:
    st.markdown("""
    #### **Untuk Pengecekan Cepat:**
    1. **Lihat bagian "Prediksi Jangka Pendek"** untuk info 6 jam ke depan
    2. **Check Sistem Peringatan Dini** untuk alert terbaru
    3. **Monitor status real-time** di bagian bawah untuk kondisi saat ini
    4. **Perhatikan delta changes** (⬆️⬇️) untuk trend naik/turun
    """)

with tab2:
    st.markdown("""
    #### **Untuk Analisis Mendalam:**
    1. **Expand "Informasi Model Prediksi"** untuk detail teknis
    2. **Analisis plot komprehensif** untuk melihat performa model
    3. **Review confidence intervals** untuk tingkat kepercayaan
    4. **Compare training vs testing performance** untuk validasi model
    """)

with tab3:
    st.markdown("""
    #### **Untuk Monitoring Geografis:**
    1. **Hover pada titik-titik stasiun** untuk detail info
    2. **Perhatikan color coding**: Hijau (Normal), Kuning (Siaga 3), Orange (Siaga 2), Merah (Siaga 1)
    3. **Zoom in/out** untuk detail geografis yang lebih baik
    4. **Use scroll zoom** untuk navigasi yang lebih mudah
    """)

st.markdown("---")

# === PAGE 2: ANALISIS DATA ===
st.subheader("📊 Halaman 2: Informasi dan Analisis Data")

st.markdown("### **Fungsi Utama:**")
st.markdown("""
Halaman analisis adalah **tool eksploratori** yang memberikan insight mendalam tentang karakteristik data time series, 
pola, korelasi, dan seasonal behavior untuk keperluan penelitian dan analisis teknis.
""")

with st.expander("🔍 Fitur-Fitur Halaman Analisis Data", expanded=True):
    st.markdown("""
    #### **1. 📤 Upload Data atau Gunakan Default**
    - **Upload custom CSV/Excel** untuk analisis data sendiri
    - **Data default** menggunakan dataset Katulampa-Depok-Manggarai
    - **Validation** format data dan requirements
    - **Error handling** untuk file yang tidak sesuai
    
    #### **2. ❓ Analisis Missing Values**
    - **Deteksi otomatis** missing data per kolom
    - **Visualisasi bar chart** jumlah missing values
    - **Table summary** untuk referensi cepat
    - **Data quality assessment**
    
    #### **3. 📋 Informasi Dasar Dataset**
    - **Metrics overview**: Total baris, kolom, periode data
    - **Data range**: Tanggal awal dan akhir dataset
    - **Quick statistics** untuk data validation
    
    #### **4. 🔗 Analisis Korelasi**
    - **Interactive heatmap** antar stasiun
    - **Color-coded correlation values** (-1 to +1)
    - **Hover details** untuk nilai exact correlation
    - **Insight relationship** antar stasiun monitoring
    
    #### **5. 📈 Plot Time Series Interaktif**
    - **Multi-line plot** semua stasiun dalam satu chart
    - **Toggle visibility** per stasiun
    - **Zoom dan pan functionality**
    - **Unified hover mode** untuk comparison
    
    #### **6. 📊 Analisis per Stasiun (Detail)**
    - **4-panel analysis**: Time series, Histogram, Boxplot, Monthly trend
    - **Distribution analysis** untuk understanding data patterns
    - **Outlier detection** melalui boxplot
    - **Monthly aggregation** untuk seasonal insights
    
    #### **7. 🔄 Cross-Correlation Analysis**
    - **Dynamic target selection** (berubah sesuai stasiun pilihan)
    - **Lag analysis** untuk menentukan optimal lag times
    - **Star markers** untuk maximum correlation points
    - **Summary table** dengan lag optimal dan correlation values
    
    #### **8. 🔄 Dekomposisi Time Series**
    - **Seasonal decomposition** (additive model)
    - **4 komponen**: Original, Trend, Seasonal, Residual
    - **Period selection**: Daily (24h) atau Weekly (168h)
    - **Interactive subplots** untuk detail analysis
    
    #### **9. 📈 Statistik Deskriptif**
    - **Comprehensive statistics**: mean, std, min, max, percentiles
    - **Cross-station comparison** dalam satu table
    - **Data profiling** untuk feature understanding
    """)

st.markdown("### Cara Menggunakan Halaman Analisis:")

tab1, tab2, tab3 = st.tabs(["🚀 Getting Started", "🔍 Exploratory Analysis", "🧪 Advanced Analysis"])

with tab1:
    st.markdown("""
    #### **Memulai Analisis:**
    1. **Choose data source**: Upload file atau gunakan default
    2. **Validate data**: Check missing values dan basic info
    3. **Review data requirements** di expandable section
    4. **Confirm data format** sesuai dengan standard yang diharapkan
    """)

with tab2:
    st.markdown("""
    #### **Analisis Eksploratori:**
    1. **Start dengan correlation heatmap** untuk understanding relationships
    2. **Review time series plot** untuk overall patterns
    3. **Select specific station** untuk detailed analysis
    4. **Analyze distribution** menggunakan histogram dan boxplot
    5. **Check monthly trends** untuk seasonal insights
    """)

with tab3:
    st.markdown("""
    #### **Analisis Lanjutan:**
    1. **Cross-correlation analysis**:
       - Pilih stasiun target dari dropdown
       - Adjust lag range sesuai kebutuhan
       - Note optimal lag untuk feature engineering
    
    2. **Seasonal decomposition**:
       - Pilih period (24h untuk daily, 168h untuk weekly)
       - Analyze trend component untuk long-term patterns
       - Review seasonal component untuk recurring patterns
       - Check residuals untuk noise/outliers
    
    3. **Statistical profiling**:
       - Compare descriptive statistics across stations
       - Identify outliers dan anomalies
       - Validate data quality untuk modeling
    """)

st.markdown("---")


show_footer()