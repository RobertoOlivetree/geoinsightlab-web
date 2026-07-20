import streamlit as st

from utils.data_loader import load_data
from utils.theme import apply_theme


apply_theme()
dataframe = load_data()

st.markdown(
    """
    <div class="geo-hero">
        <h1>GEOInsightLab</h1>
        <h3>Open Spatial Data Science Laboratory</h3>
        <p>
            An interactive research platform integrating spatial data science,
            urban accessibility, machine learning and explainable artificial intelligence
            to strengthen spatial literacy and support evidence-informed territorial
            decision-making.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    'GEOInsightLab originated from the doctoral research '
    '"Data Science in Spatial Literacy", developed at NOVA University Lisbon '
    '(NOVA FCSH / CICS.NOVA). The platform translates the scientific methods '
    'developed during the research into an interactive, transparent and '
    'reproducible environment for research, education and territorial analysis.'
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Buildings", f"{len(dataframe):,}")
metric_2.metric("Variables", len(dataframe.columns))
metric_3.metric("Parishes", dataframe["designacao_simplificada"].nunique())
metric_4.metric("Spatial unit", "Building")

st.subheader("Scientific framework")
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
            <h3>🗺️ Spatial Data Exploration</h3>
            <p>
                Explore the spatial and statistical structure of the Porto case study
                through interactive filters, maps, distributions, correlations and
                data export.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with column_2:
    st.markdown(
        """
        <div class="geo-card">
            <h3>🔬 Research Modules</h3>
            <p>
                Examine the analytical modules derived from the doctoral research,
                including spatial clustering, urban attractiveness, explainable
                artificial intelligence and spatial diagnostics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with column_3:
    st.markdown(
        """
        <div class="geo-card">
            <h3>♻️ Open and Reproducible Science</h3>
            <p>
                Methods, data-processing steps and results are organised to support
                transparency, scientific traceability, reproducibility and future
                extension of the platform.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Featured case study")
st.markdown(
    """
    **Urban Accessibility — Porto, Portugal**

    The current release is connected to the building-level dataset developed for the
    Porto case study. It provides the scientific foundation for testing spatial
    accessibility, urban attractiveness, machine-learning and spatial-diagnostic
    methods in an interactive environment.
    """
)

st.subheader("Current release")
st.success(
    "The current research edition provides an English-language interface, "
    "an expanded spatial data exploration module and the navigation structure "
    "for the progressive integration of the scientific modules developed within "
    "the doctoral research."
)

st.divider()
st.caption(
    'Developed by Roberto de Oliveira Machado · NOVA FCSH / CICS.NOVA · '
    'Originating from the doctoral research "Data Science in Spatial Literacy"'
)
