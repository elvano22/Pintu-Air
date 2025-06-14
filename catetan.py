import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configure page settings
st.set_page_config(
    page_title="Sistem Monitoring dan Prediksi Ketinggian Pintu Air Manggarai",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data():
    try:
        # Load your main dataset
        if os.path.exists('data/02 All Data.csv'):
            df_main = pd.read_csv('data/02 All Data.csv')
            df_main['Tanggal'] = pd.to_datetime(df_main['Tanggal'])
        else:
            df_main = None
            
        # Load lagged dataset
        if os.path.exists('data/06_lagged_dataset.csv'):
            df_lagged = pd.read_csv('data/06_lagged_dataset.csv')
            df_lagged['Tanggal'] = pd.to_datetime(df_lagged['Tanggal'])
        else:
            df_lagged = None
            
        return df_main, df_lagged
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Cache the data loading function
@st.cache_data
def get_cached_data():
    return load_data()

def main():
    """
    Main function for the homepage
    """
    # Header
    st.markdown('<div class="main-header">🌊 Prediksi Tinggi Muka Air Jakarta</div>', 
                unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">Time Series Forecasting untuk Data Tinggi Muka Air per Jam</div>', 
                unsafe_allow_html=True)
    
    # Introduction section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Tujuan Penelitian</h3>
        <p>Penelitian ini bertujuan untuk mengembangkan model prediksi tinggi muka air di wilayah Jakarta 
        menggunakan pendekatan time series forecasting. Model ini diharapkan dapat membantu dalam 
        sistem peringatan dini banjir dan manajemen sumber daya air.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>📊 Dataset</h3>
        <p>Data yang digunakan meliputi pengukuran tinggi muka air per jam dari berbagai stasiun 
        monitoring di Jakarta, mulai dari Oktober 2021 hingga April 2025. Stasiun monitoring meliputi:</p>
        <ul>
            <li><strong>Hulu:</strong> Katulampa, Depok</li>
            <li><strong>Tengah:</strong> Manggarai, Karet, Krukut</li>
            <li><strong>Hilir:</strong> Pesanggrahan, Angke, Pluit, Pasar Ikan, Cipinang, Sunter, Pulo Gadung</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>🔧 Metodologi</h3>
        <ul>
            <li>📈 Exploratory Data Analysis</li>
            <li>🔄 Data Preprocessing & Lagging</li>
            <li>🤖 Model Development</li>
            <li>📊 Model Evaluation</li>
            <li>🔮 Real-time Prediction</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Current datetime display
        st.markdown("""
        <div class="metric-container">
        <h4>⏰ Waktu Saat Ini</h4>
        """ + f"<p>{datetime.now().strftime('%d %B %Y, %H:%M WIB')}</p>" + """
        </div>
        """, unsafe_allow_html=True)
    
    # Data overview section
    st.markdown('<div class="sub-header">📋 Ringkasan Dataset</div>', unsafe_allow_html=True)
    
    # Load data for overview
    df_main, df_lagged = get_cached_data()
    
    if df_main is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Data Points", 
                value=f"{len(df_main):,}",
                help="Jumlah total observasi dalam dataset"
            )
        
        with col2:
            st.metric(
                label="Stasiun Monitoring", 
                value="13",
                help="Jumlah stasiun pemantauan tinggi muka air"
            )
        
        with col3:
            if 'Tanggal' in df_main.columns:
                date_range = (df_main['Tanggal'].max() - df_main['Tanggal'].min()).days
                st.metric(
                    label="Periode Data", 
                    value=f"{date_range} hari",
                    help="Rentang waktu pengumpulan data"
                )
        
        with col4:
            st.metric(
                label="Frekuensi", 
                value="Per Jam",
                help="Interval pengukuran data"
            )
        
        # Quick data preview
        with st.expander("👀 Preview Dataset Utama"):
            st.dataframe(df_main.head(10), use_container_width=True)
        
        if df_lagged is not None:
            with st.expander("🔄 Preview Dataset dengan Lag Features"):
                st.dataframe(df_lagged.head(10), use_container_width=True)
    
    else:
        st.warning("⚠️ Dataset belum tersedia. Pastikan file CSV sudah ditempatkan di folder 'data/'")
    
    # Navigation guide
    st.markdown('<div class="sub-header">🧭 Panduan Navigasi</div>', unsafe_allow_html=True)
    
    nav_col1, nav_col2 = st.columns(2)
    
    with nav_col1:
        st.markdown("""
        **📊 Exploratory Data Analysis**
        - Analisis statistik deskriptif
        - Visualisasi pola temporal
        - Korelasi antar stasiun
        - Deteksi anomali dan outliers
        """)
        
        st.markdown("""
        **🤖 Informasi Model**
        - Arsitektur model yang digunakan
        - Parameter dan hyperparameter
        - Proses training dan validasi
        - Evaluasi performa model
        """)
    
    with nav_col2:
        st.markdown("""
        **🔮 Prediksi**
        - Input data real-time
        - Hasil prediksi tinggi muka air
        - Visualisasi prediksi vs aktual
        - Confidence intervals
        """)
        
        st.markdown("""
        **💡 Tips Penggunaan**
        - Gunakan sidebar untuk navigasi
        - Hover pada grafik untuk detail
        - Download hasil untuk analisis lebih lanjut
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
    📧 Thesis Project - Time Series Forecasting for Water Level Prediction<br>
    🎓 Dikembangkan untuk keperluan akademik
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()