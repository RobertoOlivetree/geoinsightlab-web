import streamlit as st

from utils.theme import apply_theme, page_header, scientific_note

apply_theme()
page_header(
    "Methodology",
    "Scientific scope, analytical workflow and reproducibility principles.",
)

scientific_note(
    "GEOInsightLab distinguishes validated thesis reproduction from interactive exploration. "
    "This separation prevents exploratory outputs from being presented as original validated results."
)

st.subheader("Analytical workflow")
st.markdown(
    """
1. Spatial data acquisition and preprocessing
2. Building-level demographic and accessibility indicators
3. Exploratory spatial data analysis
4. Unsupervised learning and urban-profile identification
5. Supervised learning and explainable artificial intelligence
6. Accessibility and 15-minute city analysis
7. Urban attractiveness modelling
8. Spatial autocorrelation and local diagnostics
"""
)

st.subheader("Reproducibility principles")
st.markdown(
    """
- The source dataset is validated before each module is rendered.
- Model results are only included when the corresponding artefacts are available.
- Methods, parameters and assumptions are documented with each scientific module.
- Interactive reconstructions are identified as exploratory outputs.
"""
)
