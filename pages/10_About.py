import streamlit as st

from utils.theme import apply_theme, page_header

apply_theme()
page_header(
    "About",
    "Project context and scientific authorship.",
)

st.markdown(
    """
### GEOInsightLab

GEOInsightLab is an interactive research platform developed in connection with the doctoral project **Data Science in Spatial Literacy**. It translates a spatial data science workflow into accessible, transparent and reproducible interactive modules.

### Author

**Roberto de Oliveira Machado**  
Geographer and Spatial Data Scientist  
NOVA FCSH / CICS.NOVA

### Scientific scope

The platform focuses on spatial literacy, urban accessibility, spatial machine learning, explainable artificial intelligence and evidence-informed territorial decision support.
"""
)
