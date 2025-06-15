import streamlit as st
import base64

def show_footer():

    st.markdown("---")
    with open("assets/images/logo_gray.svg", "rb") as img_file:
        logo_base64 = base64.b64encode(img_file.read()).decode()

    st.markdown(f"""
    <div style="text-align: center; padding: 0;">
    <img src="data:image/svg+xml;base64,{logo_base64}" width="80" style="vertical-align: middle;">
    <span style="margin: 0 15px; font-size: 30px; color: #666666; vertical-align: middle;">|</span>
    <span style="font-size: 14px; color: #666666; vertical-align: middle;">
        <strong>Elvano Jethro Mogi Pardede</strong> - 2025
    </span>
    </div>
    """, unsafe_allow_html=True)