"""GEOInsightLab — Spatial Diagnostics (cloud-safe).

Reproduces the spatial-diagnostics workflow of the research notebooks without
constructing GeoDataFrames or dense spatial-weight matrices at runtime.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pyproj import Transformer
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

from utils.data_loader import load_data
from utils.theme import apply_theme, page_header, scientific_note


TOTAL_AGENTS = 22_000
K_NEIGHBOURS = 45
RANDOM_STATE = 42
N_PERMUTATIONS = 199  # cloud-safe; observed Moran's I is exact

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

QUADRANT_ORDER = ["High–High", "Low–Low", "High–Low", "Low–High"]


@dataclass
class Results:
    diagnostics: pd.DataFrame
    moran_attractiveness: float
    moran_agents: float
    moran_residuals: float
    moran_residuals_p: float
    metrics: dict[str, float]


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "latitude",
        "longitude",
        "osm_id",
        *ATTRACTIVENESS_WEIGHTS.keys(),
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if len(df) <= K_NEIGHBOURS:
        raise ValueError(f"At least {K_NEIGHBOURS + 1} records are required.")


def projected_coordinates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 coordinates to ETRS89 / Portugal TM06 (EPSG:3763)."""

    lon = _numeric(df, "longitude").to_numpy(dtype=np.float64)
    lat = _numeric(df, "latitude").to_numpy(dtype=np.float64)

    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (lon >= -180)
        & (lon <= 180)
        & (lat >= -90)
        & (lat <= 90)
    )
    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum())} records contain invalid coordinates."
        )

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:3763", always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    coords = np.column_stack(
        [np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)]
    )
    return coords, valid


def attractiveness(df: pd.DataFrame) -> np.ndarray:
    weighted = np.zeros(len(df), dtype=np.float64)
    for column, weight in ATTRACTIVENESS_WEIGHTS.items():
        weighted += _numeric(df, column).to_numpy(dtype=np.float64) * weight

    values = np.log1p(np.clip(weighted, 0.0, None))
    upper = float(np.quantile(values, 0.95))
    return np.clip(values, float(values.min()), upper)


def knn_weights(coords: np.ndarray, k: int = K_NEIGHBOURS) -> sparse.csr_matrix:
    """Create row-standardised directed KNN weights as a sparse matrix."""

    model = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="auto",
        metric="euclidean",
        n_jobs=1,
    )
    model.fit(coords)
    indices = model.kneighbors(coords, return_distance=False)[:, 1:]

    n = len(coords)
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = indices.reshape(-1).astype(np.int32, copy=False)
    data = np.full(n * k, 1.0 / k, dtype=np.float32)

    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n, n),
        dtype=np.float32,
    )


def moran_i(values: np.ndarray, weights: sparse.csr_matrix) -> float:
    """Calculate Global Moran's I using row-standardised sparse weights."""

    z = np.asarray(values, dtype=np.float64)
    z = z - z.mean()
    denominator = float(z @ z)
    if denominator <= 0:
        return float("nan")

    lag = weights @ z
    n = len(z)
    s0 = float(weights.sum())
    return float((n / s0) * ((z @ lag) / denominator))


def permutation_p_value(
    values: np.ndarray,
    weights: sparse.csr_matrix,
    observed_i: float,
    permutations: int = N_PERMUTATIONS,
) -> float:
    """Sequential two-sided permutation test without large temporary arrays."""

    rng = np.random.default_rng(RANDOM_STATE)
    exceedances = 0
    centred = np.asarray(values, dtype=np.float64)
    centred = centred - centred.mean()
    denominator = float(centred @ centred)
    n = len(centred)
    s0 = float(weights.sum())

    for _ in range(permutations):
        permuted = rng.permutation(centred)
        simulated = float(
            (n / s0) * ((permuted @ (weights @ permuted)) / denominator)
        )
        if abs(simulated) >= abs(observed_i):
            exceedances += 1

    return float((exceedances + 1) / (permutations + 1))


def quadrant_frame(
    residuals: np.ndarray,
    weights: sparse.csr_matrix,
) -> pd.DataFrame:
    residual_std = (residuals - residuals.mean()) / residuals.std(ddof=1)
    lag = np.asarray(weights @ residual_std, dtype=np.float64)
    lag_std = (lag - lag.mean()) / lag.std(ddof=1)

    labels = np.select(
        [
            (residual_std > 0) & (lag_std > 0),
            (residual_std < 0) & (lag_std < 0),
            (residual_std > 0) & (lag_std < 0),
            (residual_std < 0) & (lag_std > 0),
        ],
        QUADRANT_ORDER,
        default="Low–High",
    )

    return pd.DataFrame(
        {
            "standardised_residual": residual_std,
            "standardised_spatial_lag": lag_std,
            "quadrant": labels,
        }
    )


def run_workflow(df: pd.DataFrame) -> Results:
    validate_input(df)

    coords, _ = projected_coordinates(df)
    attr = attractiveness(df)

    work = pd.DataFrame(index=df.index)
    for column in MODEL_VARIABLES[:-4]:
        work[column] = _numeric(df, column)

    work["distancia_media_servicos_log"] = np.log1p(
        _numeric(df, "distancia_media_servicos")
    )
    work["pop_64_mais_log"] = np.log1p(_numeric(df, "pop_64_mais"))
    work["coord_x"] = coords[:, 0]
    work["coord_y"] = coords[:, 1]
    work = work.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    total_attr = float(attr.sum())
    if total_attr <= 0:
        raise ValueError("The attractiveness index has a non-positive sum.")

    agents = np.round((attr / total_attr) * TOTAL_AGENTS).astype(np.float64)

    x_train, x_test, y_train, y_test = train_test_split(
        work,
        agents,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.3,
        random_state=RANDOM_STATE,
        reg_alpha=1.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    raw_predictions = model.predict(work).astype(np.float64)
    test_predictions = model.predict(x_test).astype(np.float64)
    residuals = agents - raw_predictions

    weights = knn_weights(coords, K_NEIGHBOURS)

    i_attr = moran_i(attr, weights)
    i_agents = moran_i(agents, weights)
    i_residuals = moran_i(residuals, weights)
    p_residuals = permutation_p_value(
        residuals, weights, i_residuals, N_PERMUTATIONS
    )

    quadrants = quadrant_frame(residuals, weights)

    metrics = {
        "R²": float(r2_score(y_test, test_predictions)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_test, test_predictions))
        ),
        "MAE": float(mean_absolute_error(y_test, test_predictions)),
        "Bias": float(np.mean(test_predictions - y_test)),
    }

    diagnostics = pd.DataFrame(
        {
            "osm_id": df["osm_id"].astype(str).to_numpy(),
            "longitude": _numeric(df, "longitude").to_numpy(),
            "latitude": _numeric(df, "latitude").to_numpy(),
            "atratividade": attr,
            "agentes_iniciais": agents,
            "previsao_bruta": raw_predictions,
            "residuos": residuals,
            "standardised_residual": quadrants[
                "standardised_residual"
            ].to_numpy(),
            "standardised_spatial_lag": quadrants[
                "standardised_spatial_lag"
            ].to_numpy(),
            "quadrant": quadrants["quadrant"].to_numpy(),
        }
    )

    del model, weights, work, coords
    gc.collect()

    return Results(
        diagnostics=diagnostics,
        moran_attractiveness=i_attr,
        moran_agents=i_agents,
        moran_residuals=i_residuals,
        moran_residuals_p=p_residuals,
        metrics=metrics,
    )


apply_theme()
page_header(
    "Spatial Diagnostics",
    "Global spatial autocorrelation and residual diagnostics for the "
    "XGBoost model used in the research.",
)
scientific_note(
    "This cloud-safe implementation preserves the full building dataset, "
    "the original attractiveness weights, 22,000 agents, the XGBoost "
    "specification and KNN weights with 45 neighbours. It avoids full "
    "GeoDataFrames and dense weight matrices."
)

try:
    source_data = load_data()
except Exception as exc:
    st.error(f"Data loading failed: {type(exc).__name__}: {exc}")
    st.stop()

st.caption(f"Dataset loaded: {len(source_data):,} buildings.")

if "spatial_results_v2" not in st.session_state:
    st.session_state.spatial_results_v2 = None

left, right = st.columns([1, 1])
with left:
    run_clicked = st.button(
        "Run spatial diagnostics",
        type="primary",
        use_container_width=True,
    )
with right:
    clear_clicked = st.button("Clear results", use_container_width=True)

if clear_clicked:
    st.session_state.spatial_results_v2 = None
    gc.collect()
    st.rerun()

if run_clicked:
    try:
        status = st.status("Running spatial diagnostics…", expanded=True)
        status.write("1/5 — Preparing projected coordinates")
        status.write("2/5 — Calculating attractiveness and agent allocation")
        status.write("3/5 — Training XGBoost")
        status.write("4/5 — Building sparse KNN weights")
        status.write("5/5 — Calculating Moran's I and residual diagnostics")

        st.session_state.spatial_results_v2 = run_workflow(source_data)
        status.update(
            label="Spatial diagnostics completed",
            state="complete",
            expanded=False,
        )
    except Exception as exc:
        st.exception(exc)

results = st.session_state.spatial_results_v2

if results is None:
    st.info("Press **Run spatial diagnostics** to start the calculation.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Attractiveness Moran's I", f"{results.moran_attractiveness:.4f}")
m2.metric("Agents Moran's I", f"{results.moran_agents:.4f}")
m3.metric("Residual Moran's I", f"{results.moran_residuals:.4f}")
m4.metric(
    f"Residual p-value ({N_PERMUTATIONS} permutations)",
    f"{results.moran_residuals_p:.4f}",
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Global Moran's I",
        "Moran/LISA diagram",
        "Residual diagnostics",
        "Methodology",
    ]
)

with tab1:
    global_table = pd.DataFrame(
        {
            "Indicator": [
                "Urban attractiveness",
                "Initial proportional agents",
                "XGBoost residuals",
            ],
            "Moran's I": [
                results.moran_attractiveness,
                results.moran_agents,
                results.moran_residuals,
            ],
        }
    )
    chart = px.bar(
        global_table,
        x="Indicator",
        y="Moran's I",
        text_auto=".4f",
        title="Global Moran's I",
    )
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(global_table.round(4), hide_index=True, use_container_width=True)

with tab2:
    chart = px.scatter(
        results.diagnostics,
        x="standardised_residual",
        y="standardised_spatial_lag",
        color="quadrant",
        category_orders={"quadrant": QUADRANT_ORDER},
        opacity=0.55,
        hover_data={"osm_id": True, "residuos": ":.4f"},
        labels={
            "standardised_residual": "Standardised residual",
            "standardised_spatial_lag": "Standardised spatial lag",
            "quadrant": "Quadrant",
        },
    )
    chart.add_hline(y=0)
    chart.add_vline(x=0)
    st.plotly_chart(chart, use_container_width=True)
    st.caption(
        "The four classes are sign-based Moran scatterplot quadrants. "
        "They are not statistically significant Local Moran clusters."
    )

with tab3:
    histogram = px.histogram(
        results.diagnostics,
        x="residuos",
        nbins=60,
        title="Distribution of XGBoost residuals",
    )
    st.plotly_chart(histogram, use_container_width=True)

    metric_table = pd.DataFrame(
        {"Metric": results.metrics.keys(), "Value": results.metrics.values()}
    )
    st.dataframe(metric_table.round(4), hide_index=True, use_container_width=True)

with tab4:
    st.markdown(
        f"""
- Full building-level dataset.
- Original weighted attractiveness index and 95th-percentile winsorisation.
- {TOTAL_AGENTS:,} agents allocated proportionally to attractiveness.
- Original XGBoost hyperparameters.
- Row-standardised sparse KNN weights with k = {K_NEIGHBOURS}.
- Exact observed Global Moran's I.
- Sequential residual permutation test with {N_PERMUTATIONS} permutations.
- Moran scatterplot quadrants without Local Moran significance claims.

The sparse implementation produces the same observed Moran statistic as the
equivalent row-standardised KNN representation while avoiding the memory cost
of full GeoDataFrames and Libpysal weight dictionaries.
        """
    )

    st.download_button(
        "Download results",
        data=results.diagnostics.to_csv(index=False).encode("utf-8"),
        file_name="geoinsightlab_spatial_diagnostics.csv",
        mime="text/csv",
    )
