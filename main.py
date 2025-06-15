import streamlit as st

# Page config
st.set_page_config(
    page_title="aqualert",
    page_icon="assets/images/logo_partial.svg",
    layout="wide"
)

st.logo("assets/images/logo_partial.svg", size="large")

# Define pages
beranda_page = st.Page("pages/1_beranda.py", title="Beranda", icon="🏠", default=True)
prediksi_page = st.Page("pages/2_prediksi.py", title="Prediksi", icon="🔮")
info_data_page = st.Page("pages/3_informasi_data.py", title="Informasi Data", icon="📊")
info_model_page = st.Page("pages/4_informasi_model.py", title="Informasi Model", icon="🤖")

# Navigation
pg = st.navigation([
    beranda_page,
    prediksi_page, 
    info_data_page,
    info_model_page
])

# Run selected page
pg.run()