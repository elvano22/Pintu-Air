import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Sistem Monitoring dan Prediksi Ketinggian Pintu Air Manggarai",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data loading functions with caching
@st.cache_data
def load_data():
    """Load training and testing data"""
    try:
        data = pd.read_csv('02 All Data.csv')
        y_train = pd.read_csv('02 y_train.csv')
        y_test = pd.read_csv('02 y_test.csv')
        X_test = pd.read_csv('02 X_test.csv')
        
        # Convert to arrays and flatten if needed
        y_train = y_train.values.flatten() if len(y_train.shape) > 1 else y_train.values
        y_test = y_test.values.flatten() if len(y_test.shape) > 1 else y_test.values
        
        return data, y_train, y_test, X_test
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 1rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 50rem;
#     }
#     .sidebar-header {
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: #2e86ab;
#         margin-bottom: 1rem;
#     }
#     .metric-container {
#         background-color: #f0f8ff;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 5px solid #1f77b4;
#         margin: 0.5rem 0;
#     }
#     .info-box {
#         background-color: #e6f3ff;
#         padding: 1rem;
#         border-radius: 5px;
#         border: 1px solid #b3d9ff;
#         margin: 1rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown('<p class="sidebar-header">🌊 Navigasi</p>', unsafe_allow_html=True)
    
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Beranda"
    
    # Menu buttons
    if st.button("🏠 Beranda", use_container_width=True):
        st.session_state.current_page = "🏠 Beranda"
    
    if st.button("📈 Prediksi", use_container_width=True):
        st.session_state.current_page = "📈 Prediksi"
    
    if st.button("🔍 Informasi Model", use_container_width=True):
        st.session_state.current_page = "🔍 Informasi Model"
    
    # Get the selected page from session state
    selected_page = st.session_state.current_page

# Main content area
st.markdown('<h1 class="main-header">🌊 Sistem Monitoring dan Prediksi Ketinggian Pintu Air Manggarai</h1>', unsafe_allow_html=True)
st.markdown("---")


if selected_page == "🏠 Beranda":   
    # Load data first
    data, y_train, y_test, X_test = load_data()
    
    # Function to determine alert level
    def get_alert_level(location, height):
        thresholds = {
            'Katulampa': {'Siaga 3': 80, 'Siaga 2': 150, 'Siaga 1': 200},
            'Depok': {'Siaga 3': 200, 'Siaga 2': 270, 'Siaga 1': 350},
            'Manggarai': {'Siaga 3': 750, 'Siaga 2': 850, 'Siaga 1': 950}
        }
        
        if height >= thresholds[location]['Siaga 1']:
            return "🔴 Siaga 1", "error"
        elif height >= thresholds[location]['Siaga 2']:
            return "🟠 Siaga 2", "warning" 
        elif height >= thresholds[location]['Siaga 3']:
            return "🟡 Siaga 3", "info"
        else:
            return "🟢 Normal", "success"
    
    # Check if data is available
    if data is not None and len(data) > 1:
        # Get current data (last row)
        current_data = data.iloc[-1]
        prev_data = data.iloc[-2]
        
        # Extract current values and calculate deltas
        current_katulampa = current_data.get('Katulampa (air)')
        current_depok = current_data.get('Depok (air)') 
        current_manggarai = current_data.get('Manggarai (air)')
        
        delta_katulampa = current_katulampa - prev_data.get('Katulampa (air)', current_katulampa)
        delta_depok = current_depok - prev_data.get('Depok (air)', current_depok)
        delta_manggarai = current_manggarai - prev_data.get('Manggarai (air)', current_manggarai)
        
        # Get weather info
        weather_katulampa = current_data.get('Katulampa (cuaca)', 'Data tidak tersedia')
        weather_depok = current_data.get('Depok (cuaca)', 'Data tidak tersedia')
        weather_manggarai = current_data.get('Manggarai (cuaca)', 'Data tidak tersedia')
        
        # Get last updated time
        last_updated = current_data.get('Tanggal', 'Tidak diketahui')
        
if selected_page == "🏠 Beranda":   
    # Welcome Section
    st.markdown("####  Selamat Datang di sistem monitoring dan prediksi ketinggian air Sungai Ciliwung di pusat kota Jakarta, Pintu Air Manggarai!")

    # Navigation Guide
    st.markdown("---")
    st.markdown("### 🧭 Panduan Navigasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🏠 **Beranda**
        **Halaman saat ini** - Menampilkan:
        - Status real-time tinggi muka air
        - Kondisi terkini 3 stasiun monitoring
        - Alert level berdasarkan ambang batas
        - Informasi cuaca terkini
        - Peta lokasi Pintu Air Manggarai
        """)
    
    with col2:
        st.markdown("""
        #### 📈 **Prediksi**
        Halaman forecasting yang menampilkan:
        - Prediksi tinggi muka air 24 jam kedepan
        - Grafik trend historis vs prediksi
        - Model time series yang digunakan
        - Akurasi dan evaluasi model
        - Early warning system
        """)
    
    with col3:
        st.markdown("""
        #### 🔬 **Informasi Model**
        Halaman teknis yang berisi:
        - Detail metodologi forecasting
        - Parameter model yang digunakan
        - Evaluasi performa model
        - Feature engineering process
        - Technical documentation
        """)
    
    st.markdown("---")
    st.markdown("### 💡 **Catatan Penting:**")
    st.info("""
    - **Fokus utama**: Monitoring Pintu Air Manggarai sebagai titik kritis banjir Jakarta
    - **Data upstream**: Katulampa dan Depok sebagai indikator early warning
    - **Update**: Data ter-update setiap jam dari Oktober 2021 - April 2025
    - **Ambang batas**: Menggunakan standar BMKG untuk level siaga banjir
    """)
    
    st.markdown("---")
    # Load data first
    data, y_train, y_test, X_test = load_data()
    
    # Function to determine alert level
    def get_alert_level(location, height):
        thresholds = {
            'Katulampa': {'Siaga 3': 80, 'Siaga 2': 150, 'Siaga 1': 200},
            'Depok': {'Siaga 3': 200, 'Siaga 2': 270, 'Siaga 1': 350},
            'Manggarai': {'Siaga 3': 750, 'Siaga 2': 850, 'Siaga 1': 950}
        }
        
        if height >= thresholds[location]['Siaga 1']:
            return "🔴 Siaga 1", "error"
        elif height >= thresholds[location]['Siaga 2']:
            return "🟠 Siaga 2", "warning" 
        elif height >= thresholds[location]['Siaga 3']:
            return "🟡 Siaga 3", "info"
        else:
            return "🟢 Normal", "success"
    
    # Function to get threshold info for each location
    def get_threshold_info(location):
        thresholds = {
            'Katulampa': {
                'name': 'Bendung Katulampa',
                'siaga3': '80 cm',
                'siaga2': '150 cm', 
                'siaga1': '200 cm'
            },
            'Depok': {
                'name': 'Pos Depok',
                'siaga3': '200 cm',
                'siaga2': '270 cm',
                'siaga1': '350 cm'
            },
            'Manggarai': {
                'name': 'Manggarai BKB',
                'siaga3': '750 cm',
                'siaga2': '850 cm',
                'siaga1': '950 cm'
            }
        }
        return thresholds[location]
    
    # Check if data is available
    if data is not None and len(data) > 1:
        # Get current data (last row)
        current_data = data.iloc[-1]
        prev_data = data.iloc[-2]
        
        # Extract current values and calculate deltas
        current_katulampa = current_data.get('Katulampa (air)')
        current_depok = current_data.get('Depok (air)') 
        current_manggarai = current_data.get('Manggarai (air)')
        
        delta_katulampa = current_katulampa - prev_data.get('Katulampa (air)', current_katulampa)
        delta_depok = current_depok - prev_data.get('Depok (air)', current_depok)
        delta_manggarai = current_manggarai - prev_data.get('Manggarai (air)', current_manggarai)
        
        # Get weather info
        weather_katulampa = current_data.get('Katulampa (cuaca)', 'Data tidak tersedia')
        weather_depok = current_data.get('Depok (cuaca)', 'Data tidak tersedia')
        weather_manggarai = current_data.get('Manggarai (cuaca)', 'Data tidak tersedia')
        
        # Get last updated time
        last_updated = current_data.get('Tanggal', 'Tidak diketahui')
        
        # Header
        st.markdown(f"### 📊 Status Tinggi Muka Air Saat Ini")
        st.markdown(f"**🕐 Last Updated:** {last_updated}")
        st.markdown("Monitoring pintu-pintu air pada aliran Sungai Ciliwung hingga sampai ke Jakarta. Pintu Air Katulampa -> Pintu Air Depok -> Pintu Air Manggarai")
        
        # Main status display
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # Katulampa - UPSTREAM
            if current_katulampa is not None:
                alert_status, alert_type = get_alert_level('Katulampa', current_katulampa)
                threshold_info = get_threshold_info('Katulampa')
                
                st.markdown("#### 🏔️ Katulampa")
                st.markdown("*⬆️ Hulu Sungai*")
                st.metric(
                    label="Tinggi Muka Air",
                    value=f"{current_katulampa:.1f} cm",
                    delta=f"{delta_katulampa:+.1f} cm",
                    delta_color="inverse"
                )
                
                # Alert status - NO dropdown here
                if alert_type == "error":
                    st.error(alert_status)
                elif alert_type == "warning":
                    st.warning(alert_status)
                elif alert_type == "info":
                    st.info(alert_status)
                else:
                    st.success(alert_status)
                
                # Weather
                st.markdown(f"🌤️ **Cuaca:** {weather_katulampa}")
                
                # Separate info dropdown below weather
                with st.expander("ℹ️ Informasi Ambang Batas Siaga"):
                    st.markdown(f"""
                    **{threshold_info['name']} - Ambang Batas Siaga:**
                    
                    🟢 **Normal**: < 80 cm
                    
                    🟡 **Siaga 3**: {threshold_info['siaga3']}
                    
                    🟠 **Siaga 2**: {threshold_info['siaga2']}
                    
                    🔴 **Siaga 1**: {threshold_info['siaga1']}                    
                """)
            else:
                st.markdown("#### 🏔️ Katulampa")
                st.markdown("*⬆️ Hulu Sungai*")
                st.error("❌ Data tidak tersedia")
        
        with col2:
            # Depok
            if current_depok is not None:
                alert_status, alert_type = get_alert_level('Depok', current_depok)
                threshold_info = get_threshold_info('Depok')
                
                st.markdown("#### 🏙️ Depok")
                st.markdown("*↕️ Tengah Aliran*")
                st.metric(
                    label="Tinggi Muka Air",
                    value=f"{current_depok:.1f} cm",
                    delta=f"{delta_depok:+.1f} cm",
                    delta_color="inverse"
                )
                
                # Alert status - NO dropdown here
                if alert_type == "error":
                    st.error(alert_status)
                elif alert_type == "warning":
                    st.warning(alert_status)
                elif alert_type == "info":
                    st.info(alert_status)
                else:
                    st.success(alert_status)
                
                # Weather
                st.markdown(f"🌤️ **Cuaca:** {weather_depok}")
                
                # Separate info dropdown below weather
                with st.expander("ℹ️ Informasi Ambang Batas Siaga"):
                    st.markdown(f"""
                    **{threshold_info['name']} - Ambang Batas Siaga:**
                    
                    🟢 **Normal**: < 200 cm
                    
                    🟡 **Siaga 3**: {threshold_info['siaga3']}
                    
                    🟠 **Siaga 2**: {threshold_info['siaga2']}
                    
                    🔴 **Siaga 1**: {threshold_info['siaga1']}                    
                """)
            else:
                st.markdown("#### 🏙️ Depok")
                st.markdown("*↕️ Tengah Aliran*")
                st.error("❌ Data tidak tersedia")
        
        with col3:
            # Manggarai - MAIN FOCUS
            if current_manggarai is not None:
                alert_status, alert_type = get_alert_level('Manggarai', current_manggarai)
                threshold_info = get_threshold_info('Manggarai')
                
                st.markdown("#### 🌊 **MANGGARAI**")
                st.markdown("*⬇️ Pintu Air Aliran Sungai Ciliwung di Pusat Kota Jakarta*")
                st.metric(
                    label="Tinggi Muka Air",
                    value=f"{current_manggarai:.1f} cm",
                    delta=f"{delta_manggarai:+.1f} cm",
                    delta_color="inverse"
                )
                
                # Alert status - NO dropdown here
                if alert_type == "error":
                    st.error(alert_status)
                elif alert_type == "warning":
                    st.warning(alert_status)
                elif alert_type == "info":
                    st.info(alert_status)
                else:
                    st.success(alert_status)
                
                # Weather
                st.markdown(f"🌤️ **Cuaca:** {weather_manggarai}")
                
                # Separate info dropdown below weather
                with st.expander("ℹ️ Informasi Ambang Batas Siaga"):
                    st.markdown(f"""
                    **{threshold_info['name']} - Ambang Batas Siaga:**
                    
                    🟢 **Normal**: < 750 cm
                    
                    🟡 **Siaga 3**: {threshold_info['siaga3']}
                    
                    🟠 **Siaga 2**: {threshold_info['siaga2']}
                    
                    🔴 **Siaga 1**: {threshold_info['siaga1']}                    
                """)
            else:
                st.markdown("#### 🌊 **MANGGARAI** (Stasiun Utama)")
                st.error("❌ Data tidak tersedia")
        
        # Manggarai-focused section
        st.markdown("### 🎯 Fokus Prediksi: Pintu Air Manggarai")
        st.markdown("""
        Website ini dikhususkan untuk **monitoring dan prediksi** tinggi muka air di **Pintu Air Manggarai**. 
        Data dari Katulampa dan Depok ditampilkan sebagai **indikator upstream** yang mempengaruhi kondisi di Manggarai.
        """)
        
        # Google Maps Section - Focus on Manggarai
        st.markdown("### 📍 Lokasi Pintu Air Manggarai")
        
        # Create columns for better layout
        map_col1, map_col2 = st.columns([2, 1])
        
        with map_col1:
            # Google Maps embed using components.html
            st.components.v1.html("""
            <div style="width:100%; height:400px;">
                <iframe 
                    style="height:100%;width:100%;border:0;" 
                    frameborder="0" 
                    src="https://www.google.com/maps/embed/v1/place?q=pintu+air+manggarai&key=AIzaSyBFw0Qbyq9zTFTd-tUY6dZWTgaQzuU17R8&zoom=15">
                </iframe>
            </div>
            """, height=400)
        
        with map_col2:
            st.markdown("""
            **📍 Pintu Air Manggarai:**
            - **Kota**: Jakarta Selatan
            - **Aliran**: Sungai Ciliwung
            - **Update**: Setiap jam
            
            **📊 Stasiun Pendukung:**
            - **Katulampa**: Monitoring hulu
            - **Depok**: Monitoring tengah aliran
            """)
        
        

elif selected_page == "📈 Prediksi":
    st.markdown("## 📈 Water Level Forecasting")
    
    # Forecasting controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### ⚙️ Forecast Settings")
        
        selected_station = st.selectbox(
            "Select Station:",
            ["Manggarai", "Katulampa", "Depok", "Karet", "Krukut", "Pesanggrahan"]
        )
        
        forecast_horizon = st.slider(
            "Forecast Horizon (hours):",
            min_value=1,
            max_value=24,
            value=6,
            step=1
        )
        
        start_date = st.date_input(
            "Start Date:",
            value=datetime.now().date()
        )
        
        start_time = st.time_input(
            "Start Time:",
            value=datetime.now().time()
        )
        
        if st.button("🔮 Generate Forecast", type="primary"):
            st.success(f"Generating {forecast_horizon}-hour forecast for {selected_station}...")
    
    with col2:
        st.markdown("### 📊 Forecast Results")
        st.info("Configure settings and click 'Generate Forecast' to see predictions")
        
        # Placeholder for forecast chart
        st.markdown("*Forecast chart will appear here*")
        
        # Placeholder for results table
        st.markdown("### 📋 Forecast Data")
        st.markdown("*Forecast table will appear here*")

elif selected_page == "📊 Historical Analysis":
    st.markdown("## 📊 Historical Data Analysis")
    
    # Analysis controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 🔧 Analysis Settings")
        
        analysis_stations = st.multiselect(
            "Select Stations:",
            ["Manggarai", "Katulampa", "Depok", "Karet", "Krukut", "Pesanggrahan"],
            default=["Manggarai", "Katulampa"]
        )
        
        date_range = st.date_input(
            "Date Range:",
            value=[datetime(2024, 1, 1).date(), datetime.now().date()],
            max_value=datetime.now().date()
        )
        
        chart_type = st.selectbox(
            "Chart Type:",
            ["Time Series", "Correlation Heatmap", "Seasonal Patterns", "Statistical Summary"]
        )
        
        if st.button("📊 Generate Analysis", type="primary"):
            st.success("Generating historical analysis...")
    
    with col2:
        st.markdown("### 📈 Analysis Results")
        st.info("Select stations, date range, and chart type to generate analysis")
        
        # Placeholder for analysis results
        st.markdown("*Analysis charts will appear here*")

elif selected_page == "🔍 Model Performance":
    st.markdown("## 🔍 Model Performance Metrics")
    
    # Performance tabs
    tab1, tab2, tab3 = st.tabs(["📊 Accuracy Metrics", "📈 Residual Analysis", "🎯 Feature Importance"])
    
    with tab1:
        st.markdown("### 📊 Model Accuracy")
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE", "12.5 cm", "-2.1 cm")
        with col2:
            st.metric("RMSE", "18.3 cm", "-1.8 cm")
        with col3:
            st.metric("MAPE", "8.7%", "-0.5%")
        with col4:
            st.metric("R²", "0.887", "+0.023")
        
        st.markdown("### 📈 Performance Over Time")
        st.info("Model performance charts will be displayed here")
    
    with tab2:
        st.markdown("### 📈 Residual Analysis")
        st.info("Residual plots and analysis will be shown here")
    
    with tab3:
        st.markdown("### 🎯 Feature Importance")
        st.info("Feature importance analysis will be displayed here")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "🌊 Sistem Monitoring dan Prediksi Ketinggian Pintu Air Manggarai"
    "</div>", 
    unsafe_allow_html=True
)