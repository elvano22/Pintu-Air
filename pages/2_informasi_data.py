import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose
from utils.data_loader import load_main_data
import io

st.title("📊 Informasi dan Analisis Data Time Series")

# Add brief explanation at the top
st.markdown("""
**🌊 Halaman ini untuk analisis korelasi antar stasiun dalam satu aliran sungai yang sama.**

Diperlukan data dengan kolom: `[Stasiun] (air)` dan `[Stasiun] (cuaca)` untuk analisis lengkap termasuk boxplot per kondisi cuaca.
""")

st.markdown("---")

# Data upload section
st.subheader("📤 Upload Data atau Gunakan Data Default")

data_source = st.radio(
    "Pilih sumber data:",
    options=["Upload file CSV/Excel", "Gunakan data default (02 All Data.csv)"],
    index=1
)

df_final = None

if data_source == "Upload file CSV/Excel":
    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel", 
        type=['csv', 'xlsx'],
        help="File harus berisi data time series dengan kolom yang mengandung '(air)' untuk tinggi muka air"
    )
    
    if uploaded_file is not None:
        try:
            # Read file based on extension
            if uploaded_file.name.endswith('.csv'):
                df_final = pd.read_csv(uploaded_file)
            else:
                df_final = pd.read_excel(uploaded_file)
            
            st.success(f"File '{uploaded_file.name}' berhasil diupload!")
            
        except Exception as e:
            st.error(f"Error saat membaca file: {str(e)}")
    else:
        st.info("👆 Silakan upload file CSV atau Excel untuk melanjutkan analisis")

else:
    # Use default data via data_loader
    try:
        df_final = load_main_data()
        st.success("Data default berhasil dimuat!")
    except FileNotFoundError:
        st.error("❌ File '02 All Data.csv' tidak ditemukan di folder 'data/'. Pastikan file tersebut tersedia.")
    except Exception as e:
        st.error(f"❌ Error saat memuat data default: {str(e)}")

# Proceed with analysis if data is loaded
if df_final is not None:
    # Data requirements info
    with st.expander("ℹ️ Persyaratan Data", expanded=False):
        st.markdown("""
        **Untuk mendapatkan hasil analisis yang optimal, pastikan data Anda memenuhi:**
        
        1. **Format Kolom**: Harus ada kolom yang mengandung `(air)` untuk data tinggi muka air
        2. **Kolom Tanggal**: Kolom bernama `Tanggal` atau index berupa datetime
        3. **Data Hourly**: Data sebaiknya dalam format per jam untuk analisis time series yang akurat
        4. **Konsistensi**: Data harus konsisten tanpa gap waktu yang besar
        5. **Kolom Cuaca**: Kolom `(cuaca)` akan diabaikan dalam analisis ini
        
        **Contoh format yang diharapkan:**
        - Tanggal, Katulampa (air), Depok (air), Manggarai (air), dst.
        """)
    
    # Validate data structure
    numeric_columns = [col for col in df_final.columns if '(air)' in col]
    
    if len(numeric_columns) == 0:
        st.error("❌ Tidak ditemukan kolom yang mengandung '(air)'. Pastikan data Anda memiliki kolom tinggi muka air dengan format yang benar.")
    else:
        # Set datetime index if not already set
        if not isinstance(df_final.index, pd.DatetimeIndex):
            if 'Tanggal' in df_final.columns:
                df_final['Tanggal'] = pd.to_datetime(df_final['Tanggal'])
                df_final.set_index('Tanggal', inplace=True)
            elif df_final.index.name == 'Tanggal' or 'Tanggal' in str(df_final.index.name):
                df_final.index = pd.to_datetime(df_final.index)
        
        st.success(f"Data berhasil dimuat: {len(df_final)} baris, {len(df_final.columns)} kolom")
        
        # === MISSING VALUES ANALYSIS (Moved here) ===
        st.subheader("❓ Analisis Missing Values")
        
        missing_data = df_final.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure(data=[
                    go.Bar(x=missing_data.index, y=missing_data.values,
                           marker_color='lightcoral')
                ])
                
                fig.update_layout(
                    title="Missing Values per Kolom",
                    xaxis_title="Kolom",
                    yaxis_title="Jumlah Missing Values",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(missing_data.to_frame('Missing Values'), use_container_width=True)
        else:
            st.success("✅ Tidak ada missing values dalam dataset!")
        
        # Show basic info
        st.subheader("📋 Informasi Dasar Dataset")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Baris", len(df_final))
        with col2:
            st.metric("Total Kolom", len(df_final.columns))
        with col3:
            st.metric("Periode Awal", df_final.index.min().strftime('%Y-%m-%d'))
        with col4:
            st.metric("Periode Akhir", df_final.index.max().strftime('%Y-%m-%d'))
        
        clean_labels = [col.replace(' (air)', '') for col in numeric_columns]
        
        # === CORRELATION ANALYSIS ===
        st.subheader("🔗 Analisis Korelasi")
        
        # Calculate correlation matrix
        corr = df_final[numeric_columns].corr()
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=clean_labels,
            y=clean_labels,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Matriks Korelasi Tinggi Muka Air",
            xaxis_title="Stasiun",
            yaxis_title="Stasiun",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # === TIME SERIES PLOTS ===
        st.subheader("📈 Plot Time Series Interaktif")
        
        # Create interactive time series plot
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        for i, col in enumerate(numeric_columns):
            fig.add_trace(go.Scatter(
                x=df_final.index,
                y=df_final[col],
                mode='lines',
                name=col.replace(' (air)', ''),
                line=dict(color=colors[i % len(colors)], width=2),
                opacity=0.8
            ))
        
        fig.update_layout(
            title="Time Series Tinggi Muka Air - Semua Stasiun",
            xaxis_title="Tanggal",
            yaxis_title="Tinggi Muka Air (cm)",
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual station analysis
        st.subheader("📊 Analisis per Stasiun")
        selected_station = st.selectbox(
            "Pilih stasiun untuk analisis detail:",
            options=clean_labels,
            index=clean_labels.index('Manggarai') if 'Manggarai' in clean_labels else 0
        )
        
        selected_col = f"{selected_station} (air)"
        selected_weather_col = f"{selected_station} (cuaca)"
        
        # Create subplot for selected station
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Time Series', 'Distribusi (Histogram)', 
                           'Boxplot', 'Trend Bulanan'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Time series
        fig.add_trace(
            go.Scatter(x=df_final.index, y=df_final[selected_col], 
                      mode='lines', name='Tinggi Muka Air',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df_final[selected_col], nbinsx=50, 
                        name='Distribusi', marker_color='lightblue'),
            row=1, col=2
        )
        
        # Boxplot
        fig.add_trace(
            go.Box(y=df_final[selected_col], name='Boxplot', 
                   marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Monthly trend
        monthly_avg = df_final[selected_col].resample('ME').mean()
        fig.add_trace(
            go.Scatter(x=monthly_avg.index, y=monthly_avg.values,
                      mode='lines+markers', name='Rata-rata Bulanan',
                      line=dict(color='red', width=3)),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Analisis Detail - Stasiun {selected_station}",
            height=800,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # === WEATHER-BASED ANALYSIS (New separate section) ===
        st.subheader(f"🌤️ Analisis Tinggi Muka Air Berdasarkan Kondisi Cuaca - {selected_station}")
        
        # Check if weather data exists for selected station
        if selected_weather_col in df_final.columns:
            # Get unique weather conditions and clean them
            weather_data = df_final[selected_weather_col].dropna()
            unique_weather = weather_data.unique()
            
            # Filter out empty or very short strings
            unique_weather = [w for w in unique_weather if isinstance(w, str) and len(w.strip()) > 0]
            
            if len(unique_weather) > 0:
                st.markdown(f"""
                **Kondisi cuaca yang terdeteksi di stasiun {selected_station}:**  
                `{', '.join(sorted(unique_weather))}`
                """)
                
                # Create weather-based boxplot
                fig_weather = go.Figure()
                
                # Color palette for different weather conditions
                colors = px.colors.qualitative.Set3
                
                for i, weather in enumerate(sorted(unique_weather)):
                    # Filter data for this weather condition
                    weather_mask = df_final[selected_weather_col] == weather
                    weather_water_levels = df_final.loc[weather_mask, selected_col].dropna()
                    
                    if len(weather_water_levels) > 0:
                        fig_weather.add_trace(go.Box(
                            y=weather_water_levels,
                            name=weather,
                            marker_color=colors[i % len(colors)],
                            boxpoints='outliers',  # Show only outliers
                            # Remove jitter and pointpos to keep outliers in proper position
                        ))
                
                fig_weather.update_layout(
                    title=f"Distribusi Tinggi Muka Air per Kondisi Cuaca - {selected_station}",
                    xaxis_title="Kondisi Cuaca",
                    yaxis_title="Tinggi Muka Air (cm)",
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig_weather, use_container_width=True)
                
                # Weather statistics table
                st.subheader("📊 Statistik Tinggi Muka Air per Kondisi Cuaca")
                
                weather_stats = []
                for weather in sorted(unique_weather):
                    weather_mask = df_final[selected_weather_col] == weather
                    weather_water_levels = df_final.loc[weather_mask, selected_col].dropna()
                    
                    if len(weather_water_levels) > 0:
                        weather_stats.append({
                            'Kondisi Cuaca': weather,
                            'Jumlah Data': len(weather_water_levels),
                            'Rata-rata (cm)': f"{weather_water_levels.mean():.2f}",
                            'Median (cm)': f"{weather_water_levels.median():.2f}",
                            'Std Dev (cm)': f"{weather_water_levels.std():.2f}",
                            'Min (cm)': f"{weather_water_levels.min():.2f}",
                            'Max (cm)': f"{weather_water_levels.max():.2f}",
                            'Q25 (cm)': f"{weather_water_levels.quantile(0.25):.2f}",
                            'Q75 (cm)': f"{weather_water_levels.quantile(0.75):.2f}"
                        })
                
                if weather_stats:
                    weather_df = pd.DataFrame(weather_stats)
                    st.dataframe(weather_df, use_container_width=True)
                    
                    # Weather insights
                    with st.expander("💡 Insights Analisis Cuaca", expanded=False):
                        # Find weather condition with highest and lowest average
                        max_weather = max(weather_stats, key=lambda x: float(x['Rata-rata (cm)'].replace(',', '.')))
                        min_weather = min(weather_stats, key=lambda x: float(x['Rata-rata (cm)'].replace(',', '.')))
                        
                        st.markdown(f"""
                        **🔍 Key Insights:**
                        
                        - **Kondisi cuaca dengan tinggi muka air tertinggi**: `{max_weather['Kondisi Cuaca']}` 
                          (rata-rata: {max_weather['Rata-rata (cm)']} cm)
                        
                        - **Kondisi cuaca dengan tinggi muka air terendah**: `{min_weather['Kondisi Cuaca']}` 
                          (rata-rata: {min_weather['Rata-rata (cm)']} cm)
                        
                        - **Total kondisi cuaca berbeda**: {len(unique_weather)} kategori
                        
                        **🌦️ Interpretasi:**
                        - Boxplot menunjukkan distribusi, median, dan outliers untuk setiap kondisi cuaca
                        - Outliers dapat mengindikasikan kondisi ekstrem atau data anomali
                        - Perbedaan median antar kondisi cuaca dapat menunjukkan pengaruh cuaca terhadap tinggi muka air
                        
                        **💡 Tips Penggunaan:**
                        - Gunakan analisis ini untuk memahami pola cuaca-air
                        - Identifikasi kondisi cuaca yang berpotensi menyebabkan banjir
                        - Validasi data untuk memastikan kualitas pengukuran
                        """)
                
            else:
                st.warning(f"⚠️ Data cuaca untuk stasiun {selected_station} tidak memiliki nilai yang valid untuk dianalisis.")
        else:
            st.info(f"ℹ️ Data cuaca tidak tersedia untuk stasiun {selected_station}. "
                   f"Pastikan dataset memiliki kolom '{selected_weather_col}' untuk analisis cuaca.")
            
            st.markdown("""
            **📝 Format data cuaca yang diharapkan:**
            - Kolom bernama: `[Nama Stasiun] (cuaca)`
            - Contoh: `Manggarai (cuaca)`, `Depok (cuaca)`, `Katulampa (cuaca)`
            - Nilai bisa berupa: `cerah`, `hujan`, `berawan`, `hujan ringan`, `hujan lebat`, dll.
            - Sistem akan otomatis mendeteksi semua kategori cuaca yang ada dalam data
            """)
        
        # === CROSS CORRELATION ANALYSIS (Updated to use selected station) ===
        st.subheader(f"🔄 Analisis Cross-Correlation dengan {selected_station}")
        
        if selected_col in df_final.columns:
            max_lag = st.slider("Maksimum Lag (jam):", min_value=12, max_value=48, value=24)
            
            # Calculate cross-correlation with selected station
            # Include the selected station itself for autocorrelation analysis
            all_stations = [col for col in numeric_columns]
            available_stations = [col for col in all_stations if df_final[col].notna().sum() > 50]  # Only stations with sufficient data
            
            if len(available_stations) > 0:
                fig = go.Figure()
                colors = px.colors.qualitative.Set1
                
                summary_data = []
                
                for i, col in enumerate(available_stations):
                    lags = list(range(1, max_lag+1))
                    corrs = []
                    
                    for lag in lags:
                        if col == selected_col:
                            # Autocorrelation: correlate station with itself at different lags
                            shifted = df_final[col].shift(lag)
                            corr = shifted.corr(df_final[col])
                        else:
                            # Cross-correlation: correlate other stations with selected station
                            shifted = df_final[col].shift(lag)
                            corr = shifted.corr(df_final[selected_col])
                        corrs.append(corr)
                    
                    color = colors[i % len(colors)]
                    station_name = col.replace(" (air)", "")
                    
                    # Different line style for autocorrelation
                    if col == selected_col:
                        line_style = dict(color=color, width=4, dash='solid')
                        marker_style = dict(size=8, symbol='circle')
                        correlation_type = " (Autocorrelation)"
                    else:
                        line_style = dict(color=color, width=3, dash='dash')
                        marker_style = dict(size=6, symbol='diamond')
                        correlation_type = ""
                    
                    # Add line
                    fig.add_trace(go.Scatter(
                        x=lags, y=corrs,
                        mode='lines+markers',
                        name=station_name + correlation_type,
                        line=line_style,
                        marker=marker_style
                    ))
                    
                    # Find and mark maximum
                    max_idx = np.argmax(np.abs(corrs))  # Use absolute value for maximum correlation
                    max_lag_val = lags[max_idx]
                    max_corr_val = corrs[max_idx]
                    
                    fig.add_trace(go.Scatter(
                        x=[max_lag_val], y=[max_corr_val],
                        mode='markers',
                        name=f'{station_name} Max',
                        marker=dict(color=color, size=15, symbol='star'),
                        showlegend=False
                    ))
                    
                    # Add to summary
                    correlation_type_label = "Autocorrelation" if col == selected_col else "Cross-correlation"
                    summary_data.append({
                        'Stasiun': station_name,
                        'Tipe': correlation_type_label,
                        'Lag Optimal (jam)': max_lag_val,
                        'Korelasi Maksimum': f"{max_corr_val:.4f}",
                        'Korelasi Lag-1': f"{corrs[0]:.4f}" if len(corrs) > 0 else "N/A"
                    })
                
                fig.update_layout(
                    title=f"Cross-Correlation dan Autocorrelation dengan {selected_station}",
                    xaxis_title="Lag (jam)",
                    yaxis_title="Korelasi",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary table
                st.subheader("📊 Ringkasan Cross-Correlation dan Autocorrelation")
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Add explanation about autocorrelation
                with st.expander("💡 Penjelasan Autocorrelation vs Cross-Correlation", expanded=False):
                    st.markdown(f"""
                    **🔍 Perbedaan Analisis:**
                    
                    **📈 Autocorrelation ({selected_station}):**
                    - Mengukur korelasi data **{selected_station} dengan dirinya sendiri** pada lag yang berbeda
                    - Garis **solid tebal** pada plot
                    - Berguna untuk mendeteksi **pola temporal internal** dan **seasonality**
                    - Lag optimal menunjukkan **siklus berulang** dalam data (misal: pola harian, mingguan)
                    
                    **🔄 Cross-Correlation (Stasiun Lain):**
                    - Mengukur korelasi **stasiun lain dengan {selected_station}** pada lag yang berbeda
                    - Garis **dash** pada plot
                    - Berguna untuk menentukan **waktu travel** air antar stasiun
                    - Lag optimal menunjukkan **delay time** dari upstream ke downstream
                    
                    **💡 Interpretasi Praktis:**
                    - **Autocorrelation tinggi** pada lag 24: Pola harian yang kuat
                    - **Cross-correlation tinggi** Katulampa→Manggarai pada lag 12: Air butuh ~12 jam mengalir
                    - **Lag optimal** berbeda antar stasiun menunjukkan karakteristik aliran yang unik
                    
                    **🎯 Aplikasi untuk Prediksi:**
                    - Gunakan lag optimal untuk **feature engineering**
                    - Autocorrelation membantu menentukan **seasonal patterns**
                    - Cross-correlation menentukan **upstream predictors** yang relevan
                    """)
            else:
                st.info("Tidak ada stasiun dengan data yang cukup untuk analisis cross-correlation.")
        
        # === SEASONAL DECOMPOSITION (Increased height) ===
        st.subheader("🔄 Dekomposisi Time Series")
        
        if selected_col in df_final.columns:
            period = st.selectbox("Pilih periode untuk dekomposisi:", 
                                 options=[24, 168], 
                                 format_func=lambda x: f"{x} jam ({'Harian' if x==24 else 'Mingguan'})",
                                 index=0)
            
            try:
                # Perform seasonal decomposition
                decomposition = seasonal_decompose(
                    df_final[selected_col].dropna(),
                    model='additive',
                    period=period,
                    extrapolate_trend='freq'
                )
                
                # Create interactive decomposition plot with increased height
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Data Asli', 'Trend', 'Seasonal', 'Residual'),
                    vertical_spacing=0.03
                )
                
                # Original
                fig.add_trace(
                    go.Scatter(x=decomposition.observed.index, 
                              y=decomposition.observed.values,
                              mode='lines', name='Original',
                              line=dict(color='blue', width=1)),
                    row=1, col=1
                )
                
                # Trend
                fig.add_trace(
                    go.Scatter(x=decomposition.trend.index, 
                              y=decomposition.trend.values,
                              mode='lines', name='Trend',
                              line=dict(color='red', width=2)),
                    row=2, col=1
                )
                
                # Seasonal
                fig.add_trace(
                    go.Scatter(x=decomposition.seasonal.index, 
                              y=decomposition.seasonal.values,
                              mode='lines', name='Seasonal',
                              line=dict(color='green', width=1)),
                    row=3, col=1
                )
                
                # Residual
                fig.add_trace(
                    go.Scatter(x=decomposition.resid.index, 
                              y=decomposition.resid.values,
                              mode='lines', name='Residual',
                              line=dict(color='purple', width=1)),
                    row=4, col=1
                )
                
                fig.update_layout(
                    title=f"Dekomposisi Time Series - {selected_station} (periode {period} jam)",
                    height=1000,  # Increased height
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error dalam dekomposisi: {str(e)}")
                st.info("Coba dengan periode yang berbeda atau pastikan data memiliki observasi yang cukup.")
        
        # === DATA STATISTICS ===
        st.subheader("📈 Statistik Deskriptif")
        
        stats_df = df_final[numeric_columns].describe()
        st.dataframe(stats_df.round(2), use_container_width=True)

# === INSIGHTS AND INTERPRETATION ===
st.subheader("💡 Insights dan Interpretasi")

with st.expander("📖 Penjelasan Analisis", expanded=False):
    st.markdown("""
    ### 🔗 Analisis Korelasi
    - **Warna merah**: Korelasi positif kuat
    - **Warna biru**: Korelasi negatif 
    - **Hover** pada cell untuk melihat nilai eksak
    
    ### 📈 Time Series Interaktif
    - **Zoom**: Drag untuk zoom area tertentu
    - **Pan**: Shift+drag untuk menggeser
    - **Toggle**: Klik legend untuk hide/show stasiun
    
    ### 🔄 Cross-Correlation
    - **Bintang**: Menandai lag optimal
    - **Hover**: Lihat nilai korelasi detail
    - Lag tinggi normal untuk stasiun hulu
    - Analisis berubah sesuai stasiun yang dipilih
    
    ### 📊 Dekomposisi
    - **Trend**: Pola jangka panjang
    - **Seasonal**: Pola berulang harian/mingguan
    - **Residual**: Noise setelah trend dan seasonal dihilangkan
    - Berguna untuk memahami komponen-komponen dalam data time series
    
    ### 💡 Tips Navigasi
    - Semua chart bisa di-zoom dan pan
    - Double-click untuk reset view
    - Download chart dengan klik camera icon
    - Upload data sesuai format yang direkomendasikan untuk hasil optimal
    """)

# Footer
try:
    from utils.footer import show_footer
    show_footer()
except ImportError:
    pass  # Footer not available