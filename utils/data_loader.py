from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "porto_spatial_explorer.parquet"

REQUIRED_COLUMNS = {
    "osm_id",
    "type",
    "freguesia",
    "designacao_simplificada",
    "area",
    "pop_total",
    "pop_64_mais",
    "numero_servicos_proximos",
    "distancia_media_servicos",
    "Centro Saude",
    "Farmacias",
    "Hospitais",
    "Supermercados",
    "Bancos",
    "CTT",
    "Parques e jardins",
    "cluster_kmeans",
    "cluster_gmm",
    "cluster_agglo",
    "latitude",
    "longitude",
    "geometry_wkt",
}

COLUMN_LABELS = {
    "osm_id": "OSM ID",
    "type": "Building type",
    "freguesia": "Parish",
    "designacao_simplificada": "Parish",
    "area": "Building area",
    "pop_total": "Total population",
    "pop_64_mais": "Population aged 65+",
    "numero_servicos_proximos": "Nearby services",
    "distancia_media_servicos": "Mean distance to services",
    "Centro Saude": "Health centres",
    "Farmacias": "Pharmacies",
    "Hospitais": "Hospitals",
    "Supermercados": "Supermarkets",
    "Bancos": "Banks",
    "CTT": "Post offices",
    "Parques e jardins": "Parks and gardens",
    "cluster_kmeans": "K-Means cluster",
    "cluster_gmm": "GMM cluster",
    "cluster_agglo": "Agglomerative cluster",
    "latitude": "Latitude",
    "longitude": "Longitude",
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "The file 'data/porto_spatial_explorer.parquet' could not be found."
        )

    dataframe = pd.read_parquet(DATA_PATH)

    missing = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing:
        raise ValueError(
            "The dataset does not contain all required columns: " + ", ".join(missing)
        )

    return dataframe


def numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    excluded = {"latitude", "longitude"}
    return [
        column
        for column in dataframe.select_dtypes(include="number").columns.tolist()
        if column not in excluded
    ]


def filter_by_parish(
    dataframe: pd.DataFrame,
    selected_parishes: list[str],
) -> pd.DataFrame:
    if not selected_parishes:
        return dataframe.iloc[0:0].copy()

    return dataframe[
        dataframe["designacao_simplificada"].isin(selected_parishes)
    ].copy()


def column_label(column: str) -> str:
    return COLUMN_LABELS.get(column, column.replace("_", " ").title())
