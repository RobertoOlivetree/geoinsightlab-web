import streamlit as st

st.set_page_config(
    page_title="GEOInsightLab",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page("pages/01_Home.py", title="Home", icon="🏠", default=True)
data_exploration = st.Page(
    "pages/02_Data_Exploration.py", title="Data Exploration", icon="🗺️"
)
spatial_clustering = st.Page(
    "pages/03_Spatial_Clustering.py", title="Spatial Clustering", icon="🧩"
)
urban_attractiveness = st.Page(
    "pages/04_Urban_Attractiveness.py", title="Urban Attractiveness", icon="⭐"
)
explainable_ai = st.Page(
    "pages/05_Explainable_AI.py", title="Explainable AI", icon="🌳"
)
city_15 = st.Page(
    "pages/06_15_Minute_City.py", title="15-Minute City", icon="🚶"
)
shap_analysis = st.Page(
    "pages/07_SHAP_Analysis.py", title="SHAP Analysis", icon="🧠"
)
spatial_diagnostics = st.Page(
    "pages/08_Spatial_Diagnostics.py", title="Spatial Diagnostics", icon="📍"
)
methodology = st.Page(
    "pages/09_Methodology.py", title="Methodology", icon="📚"
)
about = st.Page("pages/10_About.py", title="About", icon="ℹ️")

navigation = st.navigation(
    {
        "GEOInsightLab": [home],
        "Explore": [data_exploration],
        "Scientific modules": [
            spatial_clustering,
            urban_attractiveness,
            explainable_ai,
            city_15,
            shap_analysis,
            spatial_diagnostics,
        ],
        "Research": [methodology, about],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()
