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
info_data_page = st.Page("pages/2_informasi_data.py", title="Informasi Data", icon="📊")
tutorial_page = st.Page("pages/3_tutorial.py", title="Tutorial", icon="❓")

# Navigation
pg = st.navigation([
    beranda_page,
    info_data_page,
    tutorial_page
])

# Run selected page
pg.run()