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


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Não foi encontrado o ficheiro "
            "'data/porto_spatial_explorer.parquet'."
        )

    dataframe = pd.read_parquet(DATA_PATH)

    missing = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing:
        raise ValueError(
            "O conjunto de dados não contém todas as colunas necessárias: "
            + ", ".join(missing)
        )

    return dataframe


def numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(include="number").columns.tolist()


def filter_by_parish(
    dataframe: pd.DataFrame,
    selected_parishes: list[str],
) -> pd.DataFrame:
    if not selected_parishes:
        return dataframe.iloc[0:0].copy()

    return dataframe[
        dataframe["designacao_simplificada"].isin(selected_parishes)
    ].copy()
