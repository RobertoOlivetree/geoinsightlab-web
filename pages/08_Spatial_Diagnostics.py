"""GEOInsightLab — Spatial Diagnostics.

This module reproduces the residual spatial-diagnostics workflow from the
original GEOInsightLab notebook (7_base.ipynb). It calculates the XGBoost
residuals, Global Moran's I and the LISA quadrant diagram directly from the
research dataset. No thesis image is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_data
from utils.theme import apply_theme, page_header, scientific_note

try:
    import geopandas as gpd
    from shapely import wkt
    from libpysal.weights import KNN, DistanceBand
    from esda.moran import Moran
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    SPATIAL_STACK_AVAILABLE = True
except ImportError:
    gpd = None
    wkt = None
    KNN = None
    DistanceBand = None
    Moran = None
    train_test_split = None
    XGBRegressor = None
    SPATIAL_STACK_AVAILABLE = False


apply_theme()

page_header(
    "Spatial Diagnostics",
    "Evaluate spatial autocorrelation in urban attractiveness and in the "
    "residuals of the XGBoost model used in the doctoral research.",
)

scientific_note(
    "This module reproduces the calculations implemented in the original "
    "GEOInsightLab notebook: KNN spatial weights with 45 neighbours, XGBoost "
    "residuals, Global Moran's I and the LISA quadrant diagram."
)


TOTAL_AGENTS = 22_000
K_NEIGHBOURS = 45
DISTANCE_THRESHOLD_METRES = 500
RANDOM_STATE = 42

MODEL_VARIABLES = [
    "Centro Saude",
    "Farmacias",
    "Supermercados",
    "Parques e jardins",
    "Hospitais",
    "CTT",
    "distancia_media_servicos_log",
    "pop_64_mais_log",
    "coord_x",
    "coord_y",
]

ATTRACTIVENESS_WEIGHTS = {
    "Bancos": 0.1809,
    "Centro Saude": 0.1556,
    "Farmacias": 0.1477,
    "Supermercados": 0.1636,
    "Parques e jardins": 0.1383,
    "Hospitais": 0.0520,
    "CTT": 0.1581,
    "pop_64_mais": 0.0013,
    "distancia_media_servicos": 0.0024,
}

LISA_ORDER = [
    "High–High",
    "Low–Low",
    "High–Low",
    "Low–High",
]

LISA_COLOURS = {
    "High–High": "#d7191c",
    "Low–Low": "#2c7bb6",
    "High–Low": "#fdae61",
    "Low–High": "#abd9e9",
}


@dataclass
class DiagnosticResults:
    dataframe: Any
    moran_attractiveness: Any
    moran_agents: Any
    moran_residuals_knn: Any
    moran_residuals_distance: Any
    lisa: pd.DataFrame


def winsorize_series(
    series: pd.Series,
    lower_quantile: float = 0.0,
    upper_quantile: float = 0.95,
) -> pd.Series:
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return series.clip(lower, upper)


class XGBRegressorNormalized:
    """Original normalized regressor used in the GEOInsightLab notebook."""

    def __init__(
        self,
        total_agents: int = TOTAL_AGENTS,
        atratividade_minima: float = 2.6,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.3,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.total_agents = total_agents
        self.atratividade_minima = atratividade_minima
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            reg_alpha=1.0,
            reg_lambda=1.0,
        )

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> "XGBRegressorNormalized":
        self.model.fit(features, target)
        return self

    def predict(
        self,
        features: pd.DataFrame,
        metadata: pd.DataFrame,
    ) -> np.ndarray:
        predictions = np.asarray(
            self.model.predict(features),
            dtype=float,
        )
        predictions = np.clip(
            predictions,
            0,
            None,
        )

        agents = np.zeros(
            len(predictions),
            dtype=int,
        )

        compulsory = (
            metadata["atratividade"].to_numpy()
            >= self.atratividade_minima
        )
        agents[compulsory] = 1

        remaining = (
            self.total_agents
            - int(agents.sum())
        )

        predictions[compulsory] = 0
        prediction_sum = float(
            predictions.sum()
        )

        factor = (
            remaining / prediction_sum
            if prediction_sum > 0
            else 0
        )

        scaled = predictions * factor
        floored = np.floor(scaled)
        fractions = scaled - floored

        agents += floored.astype(int)

        difference = (
            self.total_agents
            - int(agents.sum())
        )

        if difference > 0:
            indexes = np.argsort(
                -fractions
            )[:difference]
            agents[indexes] += 1
        elif difference < 0:
            indexes = np.argsort(
                fractions
            )[:abs(difference)]
            agents[indexes] -= 1

        return agents


def build_geodataframe(
    dataframe: pd.DataFrame,
) -> Any:
    required = {
        "geometry_wkt",
        *ATTRACTIVENESS_WEIGHTS.keys(),
    }
    missing = sorted(
        required.difference(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing required fields: "
            + ", ".join(missing)
        )

    prepared = dataframe.copy()
    prepared["geometry"] = prepared[
        "geometry_wkt"
    ].map(wkt.loads)

    geodataframe = gpd.GeoDataFrame(
        prepared,
        geometry="geometry",
        crs="EPSG:4326",
    )

    return geodataframe


def calculate_attractiveness(
    geodataframe: Any,
) -> None:
    attractiveness = pd.Series(
        0.0,
        index=geodataframe.index,
    )

    for field, weight in (
        ATTRACTIVENESS_WEIGHTS.items()
    ):
        values = pd.to_numeric(
            geodataframe[field],
            errors="coerce",
        ).fillna(0)

        attractiveness = (
            attractiveness
            + values * weight
        )

    attractiveness = np.log1p(
        attractiveness
    )

    geodataframe["atratividade"] = (
        winsorize_series(
            attractiveness,
            0.0,
            0.95,
        )
    )


def build_lisa_quadrants(
    residuals: pd.Series,
    weights: Any,
) -> pd.DataFrame:
    residual_std = (
        residuals - residuals.mean()
    ) / residuals.std(ddof=1)

    spatial_lag = (
        weights.sparse
        @ residual_std.to_numpy()
    )

    lag_std = (
        spatial_lag
        - spatial_lag.mean()
    ) / spatial_lag.std(ddof=1)

    quadrants = np.select(
        [
            residual_std.gt(0)
            & pd.Series(
                lag_std,
                index=residual_std.index,
            ).gt(0),
            residual_std.lt(0)
            & pd.Series(
                lag_std,
                index=residual_std.index,
            ).lt(0),
            residual_std.gt(0)
            & pd.Series(
                lag_std,
                index=residual_std.index,
            ).lt(0),
            residual_std.lt(0)
            & pd.Series(
                lag_std,
                index=residual_std.index,
            ).gt(0),
        ],
        LISA_ORDER,
        default="Low–High",
    )

    return pd.DataFrame(
        {
            "Standardised residual": (
                residual_std
            ),
            "Standardised spatial lag": (
                lag_std
            ),
            "Quadrant": quadrants,
        },
        index=residuals.index,
    )


@st.cache_data(show_spinner=False)
def run_spatial_diagnostics(
    dataframe: pd.DataFrame,
) -> DiagnosticResults:
    geodataframe = build_geodataframe(
        dataframe
    )
    projected = geodataframe.to_crs(
        epsg=3763
    )

    calculate_attractiveness(
        geodataframe
    )

    geodataframe[
        "distancia_media_servicos_log"
    ] = np.log1p(
        pd.to_numeric(
            geodataframe[
                "distancia_media_servicos"
            ],
            errors="coerce",
        ).fillna(0)
    )

    geodataframe[
        "pop_64_mais_log"
    ] = np.log1p(
        pd.to_numeric(
            geodataframe[
                "pop_64_mais"
            ],
            errors="coerce",
        ).fillna(0)
    )

    centroids = projected.geometry.centroid
    geodataframe["coord_x"] = centroids.x
    geodataframe["coord_y"] = centroids.y

    features = (
        geodataframe[
            MODEL_VARIABLES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0)
    )

    if "agentes" not in geodataframe:
        attractiveness_sum = float(
            geodataframe[
                "atratividade"
            ].sum()
        )

        geodataframe["proporcao"] = (
            geodataframe[
                "atratividade"
            ]
            / attractiveness_sum
        )

        geodataframe["agentes_iniciais"] = (
            geodataframe["proporcao"]
            * TOTAL_AGENTS
        ).round()

    else:
        geodataframe[
            "agentes_iniciais"
        ] = pd.to_numeric(
            geodataframe["agentes"],
            errors="coerce",
        ).fillna(0)

    target = geodataframe[
        "agentes_iniciais"
    ]

    x_train, _, y_train, _ = (
        train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
    )

    weights_knn = KNN.from_dataframe(
        projected,
        k=K_NEIGHBOURS,
    )
    weights_knn.transform = "r"

    moran_attractiveness = Moran(
        geodataframe[
            "atratividade"
        ].to_numpy(),
        weights_knn,
    )

    moran_agents = Moran(
        target.to_numpy(),
        weights_knn,
    )

    model = XGBRegressorNormalized()
    model.fit(
        x_train,
        y_train,
    )

    geodataframe["agentes"] = (
        model.predict(
            features,
            geodataframe,
        )
    )

    raw_predictions = (
        model.model.predict(
            features
        )
    )

    geodataframe["residuos"] = (
        target.to_numpy()
        - raw_predictions
    )

    moran_residuals_knn = Moran(
        geodataframe[
            "residuos"
        ].to_numpy(),
        weights_knn,
    )

    weights_distance = (
        DistanceBand.from_dataframe(
            projected,
            threshold=(
                DISTANCE_THRESHOLD_METRES
            ),
            silence_warnings=True,
        )
    )
    weights_distance.transform = "r"

    moran_residuals_distance = Moran(
        geodataframe[
            "residuos"
        ].to_numpy(),
        weights_distance,
    )

    lisa = build_lisa_quadrants(
        geodataframe[
            "residuos"
        ],
        weights_knn,
    )

    geodataframe[
        "lisa_quadrant"
    ] = lisa["Quadrant"]

    return DiagnosticResults(
        dataframe=geodataframe,
        moran_attractiveness=(
            moran_attractiveness
        ),
        moran_agents=moran_agents,
        moran_residuals_knn=(
            moran_residuals_knn
        ),
        moran_residuals_distance=(
            moran_residuals_distance
        ),
        lisa=lisa,
    )


if not SPATIAL_STACK_AVAILABLE:
    st.error(
        "This module requires geopandas, shapely, "
        "libpysal, esda, scikit-learn and xgboost."
    )
    st.stop()


try:
    data = load_data()
except Exception as error:
    st.error(
        f"The research dataset could not be loaded: {error}"
    )
    st.stop()


with st.spinner(
    "Calculating XGBoost residuals and spatial diagnostics..."
):
    try:
        results = run_spatial_diagnostics(
            data
        )
    except Exception as error:
        st.error(
            f"The spatial diagnostics could not be calculated: {error}"
        )
        st.stop()


metric_columns = st.columns(4)

metric_columns[0].metric(
    "Attractiveness Moran's I",
    f"{results.moran_attractiveness.I:.3f}",
)

metric_columns[1].metric(
    "Residual Moran's I",
    f"{results.moran_residuals_knn.I:.3f}",
)

metric_columns[2].metric(
    "Permutation p-value",
    f"{results.moran_residuals_knn.p_sim:.3f}",
)

metric_columns[3].metric(
    "KNN neighbours",
    str(K_NEIGHBOURS),
)


(
    lisa_tab,
    global_tab,
    residual_tab,
    sensitivity_tab,
    methodology_tab,
) = st.tabs(
    [
        "LISA diagram",
        "Global Moran's I",
        "Residual statistics",
        "Sensitivity analysis",
        "Methodology",
    ]
)


with lisa_tab:
    st.subheader(
        "LISA Diagram of Standardised Residuals"
    )

    figure = px.scatter(
        results.lisa,
        x="Standardised residual",
        y="Standardised spatial lag",
        color="Quadrant",
        category_orders={
            "Quadrant": LISA_ORDER,
        },
        color_discrete_map=(
            LISA_COLOURS
        ),
        opacity=0.65,
        title=(
            "Standardised Residuals and "
            "Neighbouring Residuals"
        ),
    )

    figure.add_hline(
        y=0,
        line_width=1,
    )
    figure.add_vline(
        x=0,
        line_width=1,
    )

    figure.update_layout(
        legend_title_text=(
            "LISA quadrant"
        ),
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )

    counts = (
        results.lisa["Quadrant"]
        .value_counts()
        .reindex(
            LISA_ORDER,
            fill_value=0,
        )
        .rename_axis(
            "LISA quadrant"
        )
        .reset_index(
            name="Buildings"
        )
    )

    st.dataframe(
        counts,
        width="stretch",
        hide_index=True,
    )


with global_tab:
    st.subheader(
        "Global Spatial Autocorrelation"
    )

    comparison = pd.DataFrame(
        {
            "Indicator": [
                "Urban attractiveness",
                "Initial agents",
                "XGBoost residuals — KNN",
                "XGBoost residuals — 500 m distance band",
            ],
            "Moran's I": [
                results.moran_attractiveness.I,
                results.moran_agents.I,
                results.moran_residuals_knn.I,
                results.moran_residuals_distance.I,
            ],
            "p-value": [
                results.moran_attractiveness.p_sim,
                results.moran_agents.p_sim,
                results.moran_residuals_knn.p_sim,
                results.moran_residuals_distance.p_sim,
            ],
        }
    )

    chart = px.bar(
        comparison,
        x="Indicator",
        y="Moran's I",
        text="Moran's I",
        title=(
            "Global Moran's I by Indicator"
        ),
    )

    chart.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
    )

    st.plotly_chart(
        chart,
        width="stretch",
    )

    st.dataframe(
        comparison.round(4),
        width="stretch",
        hide_index=True,
    )


with residual_tab:
    st.subheader(
        "XGBoost Residual Statistics"
    )

    residuals = results.dataframe[
        "residuos"
    ]

    histogram = px.histogram(
        residuals.to_frame(
            name="Residual"
        ),
        x="Residual",
        nbins=50,
        title=(
            "Distribution of XGBoost Residuals"
        ),
    )

    st.plotly_chart(
        histogram,
        width="stretch",
    )

    statistics = pd.DataFrame(
        {
            "Statistic": [
                "Observations",
                "Mean",
                "Standard deviation",
                "Minimum",
                "Median",
                "Maximum",
            ],
            "Value": [
                len(residuals),
                residuals.mean(),
                residuals.std(),
                residuals.min(),
                residuals.median(),
                residuals.max(),
            ],
        }
    )

    st.dataframe(
        statistics,
        width="stretch",
        hide_index=True,
    )


with sensitivity_tab:
    st.subheader(
        "Spatial-Weights Sensitivity"
    )

    sensitivity = pd.DataFrame(
        {
            "Spatial weights": [
                f"KNN ({K_NEIGHBOURS})",
                (
                    f"Distance band "
                    f"({DISTANCE_THRESHOLD_METRES} m)"
                ),
            ],
            "Moran's I": [
                results.moran_residuals_knn.I,
                results.moran_residuals_distance.I,
            ],
            "p-value": [
                results.moran_residuals_knn.p_sim,
                results.moran_residuals_distance.p_sim,
            ],
        }
    )

    chart = px.bar(
        sensitivity,
        x="Spatial weights",
        y="Moran's I",
        text="Moran's I",
        title=(
            "Residual Moran's I Across "
            "Spatial-Weights Specifications"
        ),
    )

    chart.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
    )

    st.plotly_chart(
        chart,
        width="stretch",
    )

    st.dataframe(
        sensitivity.round(4),
        width="stretch",
        hide_index=True,
    )


with methodology_tab:
    st.subheader(
        "Original Notebook Workflow"
    )

    st.markdown(
        """
This page follows the calculations implemented in the original
`7_base.ipynb` notebook:

1. Urban attractiveness is calculated from the weighted service indicators,
   population aged 65 or over and mean distance to services.
2. The attractiveness score is log-transformed and winsorised at the 95th
   percentile.
3. Model coordinates are calculated from building centroids in EPSG:3763.
4. A total of 22,000 initial agents is distributed proportionally to
   attractiveness.
5. XGBoost is fitted with 100 estimators, maximum depth 3, learning rate 0.3,
   L1 and L2 regularisation equal to 1.0, and random state 42.
6. Residuals are calculated as the initial agent allocation minus the raw
   XGBoost prediction.
7. Global Moran's I is calculated using KNN weights with 45 neighbours and a
   500-metre distance-band alternative.
8. The LISA diagram plots standardised residuals against the standardised
   spatial lag of neighbouring residuals.

The quadrant labels describe the signs of each observation and its spatial
lag. They are not Local Moran significance classes because the original
notebook classifies quadrants without applying a local permutation
significance test.
        """
    )

    download = results.dataframe[
        [
            "osm_id",
            "atratividade",
            "agentes_iniciais",
            "agentes",
            "residuos",
            "lisa_quadrant",
        ]
    ].copy()

    st.download_button(
        "Download spatial-diagnostics results",
        data=download.to_csv(
            index=False
        ).encode(
            "utf-8"
        ),
        file_name=(
            "geoinsightlab_spatial_diagnostics.csv"
        ),
        mime="text/csv",
    )
