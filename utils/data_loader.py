import streamlit as st
import pandas as pd

@st.cache_data
def load_main_data():
    return pd.read_csv("data/02 All Data.csv")

@st.cache_data
def load_info_data():
    return pd.read_csv("data/informasi_pintu_air.csv")

@st.cache_data
def create_master_station_data():
    info_df = load_info_data()
    water_data = load_main_data()
    
    # Get current and previous data
    current_data = water_data.iloc[-1]
    prev_data = water_data.iloc[-2]
    
    stations = []
    for _, station in info_df.iterrows():
        location = station['Lokasi']

        current_level = current_data.get(f'{location} (air)')
        prev_level = prev_data.get(f'{location} (air)')
        weather = current_data.get(f'{location} (cuaca)', 'Data tidak tersedia')
        
        # Calculate delta
        if current_level is not None and prev_level is not None:
            delta = current_level - prev_level
        else:
            delta = 0
        
        # Determine status
        if current_level is not None:
            if current_level >= station['Siaga 1']:
                status = 'Siaga 1'
            elif current_level >= station['Siaga 2']:
                status = 'Siaga 2'
            elif current_level >= station['Siaga 3']:
                status = 'Siaga 3'
            else:
                status = 'Normal'
        else:
            status = 'No Data'
        
        stations.append({
            'name': location,
            'latitude': station['Latitude'],
            'longitude': station['Longitude'],
            'current_level': current_level,
            'prev_level': prev_level,
            'delta': delta,
            'weather': weather,
            'status': status,
            'siaga_1': station['Siaga 1'],
            'siaga_2': station['Siaga 2'],
            'siaga_3': station['Siaga 3']
        })
    
    return pd.DataFrame(stations)

def get_station_data(location):
    master_data = create_master_station_data()
    return master_data[master_data['name'] == location].iloc[0]