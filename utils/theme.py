import streamlit as st


def apply_theme():
    st.set_page_config(
        page_title="GEOInsightLab",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def page_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def coming_soon(title: str):
    st.title(title)
    st.info("🚧 This scientific module is currently under development.")
    st.markdown("""
This module will be available in a future GEOInsightLab release.

Planned features include:

- Interactive visualisations
- Scientific methodology
- Spatial analysis
- Maps
- Reproducible results
""")
