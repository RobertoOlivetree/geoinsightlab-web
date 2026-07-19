import streamlit as st

from utils.data_loader import load_data
from utils.theme import apply_theme


apply_theme()
dataframe = load_data()

st.markdown(
    """
    <div class="geo-hero">
        <h1>GEOInsightLab</h1>
        <h3>Spatial Data Science for Spatial Literacy</h3>
        <p>
            An interactive scientific platform for exploring how spatial data science,
            urban accessibility, machine learning and explainable artificial intelligence
            can support spatial literacy and evidence-informed decision-making.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "This release establishes a stable technical foundation and connects the "
    "platform to the Porto case-study dataset used in the underlying doctoral research."
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Buildings", f"{len(dataframe):,}")
metric_2.metric("Variables", len(dataframe.columns))
metric_3.metric("Parishes", dataframe["designacao_simplificada"].nunique())
metric_4.metric("Spatial unit", "Building")

st.subheader("Research workflow")
st.markdown(
    """
    <div class="geo-workflow">
        Spatial Data → Data Exploration → Spatial Clustering → Urban Attractiveness →
        Explainable AI → 15-Minute City → SHAP Analysis → Spatial Diagnostics
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Explore the platform")
column_1, column_2, column_3 = st.columns(3)

with column_1:
    st.markdown(
        """
        <div class="geo-card">
            <h3>🗺️ Data Exploration</h3>
            <p>
                Explore the spatial and statistical structure of 31,873 buildings through
                interactive filters, maps, distributions, correlations and data export.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with column_2:
    st.markdown(
        """
        <div class="geo-card">
            <h3>🔬 Scientific Modules</h3>
            <p>
                Follow the research workflow from clustering and urban attractiveness to
                explainable machine learning, SHAP and spatial diagnostics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with column_3:
    st.markdown(
        """
        <div class="geo-card">
            <h3>♻️ Reproducible Research</h3>
            <p>
                Methods and results are separated from exploratory functionality to preserve
                transparency, scientific traceability and reproducibility.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Current release")
st.success(
    "Version 0.2 provides an English-language interface, a redesigned home page, "
    "an expanded Data Exploration module and the navigation structure for future scientific modules."
)

st.divider()
st.caption("Roberto de Oliveira Machado · NOVA FCSH / CICS.NOVA")
