import streamlit as st


def page_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def coming_soon(title: str):
    st.title(title)
    st.info("🚧 This scientific module is currently under development.")
    st.markdown(
        """
This module will be available in the next GEOInsightLab release.

Planned features include:

- Interactive visualisations
- Scientific methodology
- Maps
- Statistical analysis
- Reproducible results
"""
    )
