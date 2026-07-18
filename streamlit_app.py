import streamlit as st

st.set_page_config(
    page_title="GEOInsightLab",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page(
    "pages/01_Home.py",
    title="Home",
    icon="🏠",
    default=True,
)
data_exploration = st.Page(
    "pages/02_Data_Exploration.py",
    title="Data Exploration",
    icon="🗺️",
)

navigation = st.navigation(
    {
        "GEOInsightLab": [home],
        "Scientific workflow": [data_exploration],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()
