import streamlit as st

from utils.theme import apply_theme, page_header

apply_theme()

page_header(
    "About",
    "Scientific background, research context and authorship.",
)

st.markdown(
    """
## GEOInsightLab

GEOInsightLab is an open research platform that brings together spatial data science, geographic information systems (GIS), explainable artificial intelligence (XAI) and spatial statistics within an interactive and reproducible environment.

The platform was designed to support the exploration, interpretation and communication of spatial analytical workflows through transparent methodologies and interactive visualisations. Rather than presenting static outputs, GEOInsightLab allows users to examine how spatial indicators are produced, how machine learning models are developed and how spatial patterns can be interpreted using explainable artificial intelligence techniques.

Its primary objectives are to promote reproducible research, facilitate methodological transparency and contribute to evidence-informed territorial analysis across research, education and professional practice.

---

## Doctoral Research

GEOInsightLab was developed as part of the doctoral research **Data Science in Spatial Literacy**, completed at **NOVA University Lisbon**.

The research proposes an integrated framework combining Geographic Information Systems, spatial statistics, machine learning and explainable artificial intelligence to improve spatial literacy and support territorial decision-making. The platform represents the practical implementation of the methodological framework developed throughout the doctoral research.

**Doctoral Thesis**

http://hdl.handle.net/10362/201445

---

## Research Profile

The project is developed within the framework of **CICS.NOVA – Interdisciplinary Centre of Social Sciences**, where the author conducts research in spatial data science, urban accessibility, spatial modelling and explainable artificial intelligence applied to territorial planning.

**CICS.NOVA Research Profile**

https://www.cics.nova.fcsh.unl.pt/investigador/roberto-machado-3/

---

## Author

**Roberto de Oliveira Machado**

Geographer and Spatial Data Scientist

PhD in Geography and Regional Planning

CICS.NOVA – Interdisciplinary Centre of Social Sciences

NOVA University Lisbon

---

## Research Interests

- Spatial Data Science
- Geographic Information Systems (GIS)
- Spatial Machine Learning
- Explainable Artificial Intelligence (XAI)
- Urban Accessibility Analysis
- Spatial Statistics
- Spatial Decision Support Systems
- Evidence-based Territorial Planning
- Open Spatial Data
- Reproducible Geospatial Research

---

## Citation

If GEOInsightLab contributes to your research, teaching or professional activities, please consider citing the associated doctoral thesis.

For questions, collaborations or academic enquiries, please use the contact information available through the CICS.NOVA research profile.
"""
)
