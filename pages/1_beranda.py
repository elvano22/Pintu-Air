import streamlit as st
from utils.data_loader import load_main_data, load_info_data, create_master_station_data, get_station_data
import pandas as pd
import numpy as np
import base64
import plotly.express as px
from utils.footer import show_footer

# Cache
data = load_main_data()

with open("assets/images/logo_full.svg", "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

st.markdown(f"""
<div style="text-align: center; padding: 0 !important">
    <img src="data:image/svg+xml;base64,{logo_base64}" width="400">
    <h1>Pintu Air Manggarai</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("Website ini dikhususkan untuk **monitoring dan prediksi** tinggi muka air di **Pintu Air Manggarai**. Data dari Katulampa dan Depok ditampilkan sebagai **indikator upstream** yang mempengaruhi kondisi di Manggarai.")
st.markdown("**Pintu Air Manggarai** berada pada aliran **Sungai Ciliwung**. Terdapat beberapa pintu air di sepanjang aliran Sungai Ciliwung. Dimulai dari paling hulu sungai, yaitu Pintu Air Katulampa -> Pintu Air Depok -> **Pintu Air Manggarai**")

def get_alert_level(location, height):
    station = get_station_data(location)
    
    if height >= station['siaga_1']:
        return "🔴 Siaga 1", "error"
    elif height >= station['siaga_2']:
        return "🟠 Siaga 2", "warning" 
    elif height >= station['siaga_3']:
        return "🟡 Siaga 3", "info"
    else:
        return "🟢 Normal", "success"
    
# Function to get threshold info for each location
def get_threshold_info(location):

    df = load_info_data()
    row = df[df['Lokasi'] == location].iloc[0]
    
    return {
        'name': row['Lokasi'],
        'siaga3': f"{row['Siaga 3']} cm",
        'siaga2': f"{row['Siaga 2']} cm", 
        'siaga1': f"{row['Siaga 1']} cm",
        'latitude': row['Latitude'],
        'longitude': row['Longitude']
    }
    


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
st.markdown(f"## Status Tinggi Muka Air Saat Ini")
st.markdown(f"**🕐 Last Updated:** {last_updated}")

# Create plotly map with hover details
fig = px.scatter_mapbox(
    create_master_station_data(),
    lat='latitude',
    lon='longitude',
    hover_data={
        'current_level': ':.1f',
        'weather': True,
        'status': True,
    },
    hover_name='name',
    color='status',
    color_discrete_map={
        'Siaga 1': 'red',
        'Siaga 2': 'orange', 
        'Siaga 3': 'yellow',
        'Normal': 'green'
    },
    text='name',
    size_max=25,
    zoom=8,
    height=600,
    mapbox_style="carto-positron"
)
fig.update_traces(textposition="top center")
fig.update_layout(
    mapbox=dict(center=dict(lat=-6.2, lon=106.8))
)

config = {
    'scrollZoom': True,
    'doubleClick': 'reset',
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}

st.plotly_chart(fig, use_container_width=True, config=config)

# Main status display
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    # Katulampa - UPSTREAM
    if current_katulampa is not None:
        alert_status, alert_type = get_alert_level('Katulampa', current_katulampa)
        threshold_info = get_threshold_info('Katulampa')
        
        st.markdown("### Katulampa")
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
            
            🟢 **Normal**: < {threshold_info['siaga3']}
            
            🟡 **Siaga 3**: {threshold_info['siaga3']}
            
            🟠 **Siaga 2**: {threshold_info['siaga2']}
            
            🔴 **Siaga 1**: {threshold_info['siaga1']}                    
        """)
    else:
        st.markdown("### Katulampa")
        st.markdown("*⬆️ Hulu Sungai*")
        st.error("Data tidak tersedia")

with col2:
    # Depok
    if current_depok is not None:
        alert_status, alert_type = get_alert_level('Depok', current_depok)
        threshold_info = get_threshold_info('Depok')
        
        st.markdown("### Depok")
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
            
            🟢 **Normal**: < {threshold_info['siaga3']}
            
            🟡 **Siaga 3**: {threshold_info['siaga3']}
            
            🟠 **Siaga 2**: {threshold_info['siaga2']}
            
            🔴 **Siaga 1**: {threshold_info['siaga1']}                    
        """)
    else:
        st.markdown("### Depok")
        st.markdown("*↕️ Tengah Aliran*")
        st.error("Data tidak tersedia")

with col3:
    # Manggarai - MAIN FOCUS
    if current_manggarai is not None:
        alert_status, alert_type = get_alert_level('Manggarai', current_manggarai)
        threshold_info = get_threshold_info('Manggarai')
        
        st.markdown("### **MANGGARAI**")
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
            
            🟢 **Normal**: < {threshold_info['siaga3']}
            
            🟡 **Siaga 3**: {threshold_info['siaga3']}
            
            🟠 **Siaga 2**: {threshold_info['siaga2']}
            
            🔴 **Siaga 1**: {threshold_info['siaga1']}                    
        """)
    else:
        st.markdown("### **MANGGARAI**")
        st.error("Data tidak tersedia")


show_footer()
