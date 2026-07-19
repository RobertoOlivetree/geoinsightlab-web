# GEOInsightLab Web

GEOInsightLab is an interactive scientific platform associated with the doctoral research project **Data Science in Spatial Literacy**.

## Version 0.2

This release provides:

- a fully English-language interface;
- a redesigned professional home page;
- an expanded Data Exploration module;
- interactive maps, distributions, scatter plots and a correlation matrix;
- descriptive statistics and filtered CSV export;
- a scalable navigation structure for the scientific modules;
- methodology and project information pages;
- explicit separation between validated thesis reproduction and interactive exploration.

## Main Streamlit file

```text
streamlit_app.py
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Data

The application expects the following file:

```text
data/porto_spatial_explorer.parquet
```

## Scientific roadmap

Planned modules include Spatial Clustering, Urban Attractiveness, Explainable AI, the 15-Minute City, SHAP Analysis and Spatial Diagnostics. Each module will be activated only after its data, model artefacts and validation workflow have been integrated and tested.
