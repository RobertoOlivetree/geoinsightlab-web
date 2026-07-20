"""GEOInsightLab — SHAP Analysis Streamlit page.

This module follows Code 12 and Figures 61–64 of the doctoral thesis.
Spatial autocorrelation of residuals is reserved for Spatial Diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import column_label, load_data
from utils.theme import apply_theme, page_header, scientific_note

try:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None
    train_test_split = None
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

# SHAP contributions are calculated with XGBoost's native
# pred_contribs implementation. This avoids version-specific incompatibilities
# between SHAP and XGBoost while retaining exact TreeSHAP contributions.


apply_theme()

page_header(
    "SHAP Analysis",
    "Interpret the XGBoost model used in the doctoral research to estimate "
    "the distribution of agents across buildings in Porto.",
)

scientific_note(
    "This module reproduces the supervised modelling and SHAP workflow defined "
    "in Code 12 of the thesis. It focuses on global variable importance, the SHAP "
    "summary and model residuals without repeating the PCA, clustering or LIME analyses."
)


BUILD_ID = "2026-07-20-no-streamlit-cache"

RANDOM_STATE = 42
TOTAL_AGENTS = 22_000
ATTRACTIVENESS_THRESHOLD = 2.6

DISPLAY_LABELS = {
    "distancia_media_servicos_log": "Mean distance to services (log)",
    "Farmacias": "Pharmacies",
    "Supermercados": "Supermarkets",
    "coord_y": "Coordinate Y",
    "pop_64_mais_log": "Population aged 65+ (log)",
    "coord_x": "Coordinate X",
    "Parques ou jardins": "Parks and gardens",
    "Parques e jardins": "Parks and gardens",
    "Centro Saude": "Health centres",
    "CTT": "Post offices",
    "Hospitais": "Hospitals",
}

THESIS_WEIGHTS = {
    "Supermercados": 0.1614,
    "Bancos": 0.1522,
    "Farmacias": 0.1504,
    "CTT": 0.1499,
    "parks": 0.1323,
    "Centro Saude": 0.1292,
    "Hospitais": 0.0876,
    "pop_64_mais": 0.0322,
    "prox_servicos": 0.0049,
}


@dataclass
class ModelResults:
    model: Any
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: np.ndarray
    residuals: np.ndarray
    r2: float
    rmse: float
    mae: float


class XGBRegressorNormalized:
    """XGBoost regressor with the thesis threshold and total-agent normalisation."""

    def __init__(
        self,
        total_agents: int = TOTAL_AGENTS,
        atratividade_minima: float = ATTRACTIVENESS_THRESHOLD,
        **model_parameters: Any,
    ) -> None:
        self.total_agents = int(total_agents)
        self.atratividade_minima = float(atratividade_minima)
        self.model = XGBRegressor(**model_parameters)

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
    ) -> np.ndarray:
        raw = np.asarray(
            self.model.predict(features),
            dtype=float,
        )
        raw = np.clip(raw, 0.0, None)

        if "atratividade_escala_tese" in features.columns:
            raw = np.where(
                features["atratividade_escala_tese"].to_numpy()
                >= self.atratividade_minima,
                raw,
                0.0,
            )

        total = float(raw.sum())
        if total <= 0:
            return np.zeros(len(raw), dtype=float)

        return raw * (self.total_agents / total)


def display_label(field: str) -> str:
    return DISPLAY_LABELS.get(field, column_label(field))


def resolve_park_field(dataframe: pd.DataFrame) -> str:
    for candidate in (
        "Parques ou jardins",
        "Parques e jardins",
    ):
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        "The dataset must contain 'Parques ou jardins' or "
        "'Parques e jardins'."
    )


def numeric_series(
    dataframe: pd.DataFrame,
    field: str,
) -> pd.Series:
    values = pd.to_numeric(
        dataframe[field],
        errors="coerce",
    )
    median = values.median()
    return values.fillna(
        0.0 if pd.isna(median) else median
    )


def minmax_scale(values: pd.Series) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())
    value_range = maximum - minimum

    if value_range <= 0:
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
            dtype=float,
        )

    return (values - minimum) / value_range


def calculate_thesis_attractiveness(
    dataframe: pd.DataFrame,
    park_field: str,
) -> pd.Series:
    required = [
        "Bancos",
        "Centro Saude",
        "Farmacias",
        "Supermercados",
        park_field,
        "Hospitais",
        "CTT",
        "pop_64_mais",
        "distancia_media_servicos",
    ]

    missing = sorted(
        set(required).difference(dataframe.columns)
    )
    if missing:
        raise ValueError(
            "The attractiveness score cannot be reconstructed because "
            "these fields are missing: "
            + ", ".join(missing)
        )

    prepared = pd.DataFrame(
        {
            field: numeric_series(dataframe, field)
            for field in required
        },
        index=dataframe.index,
    )

    for field in prepared.columns:
        lower = prepared[field].quantile(0.01)
        upper = prepared[field].quantile(0.99)
        prepared[field] = prepared[field].clip(
            lower=lower,
            upper=upper,
        )

    prepared["prox_servicos"] = (
        -prepared.pop("distancia_media_servicos")
    )

    means = prepared.mean(axis=0)
    deviations = prepared.std(
        axis=0,
        ddof=0,
    ).replace(0, 1)

    standardised = (
        prepared - means
    ) / deviations

    weight_values = []
    for field in standardised.columns:
        if field == park_field:
            weight_values.append(
                THESIS_WEIGHTS["parks"]
            )
        else:
            weight_values.append(
                THESIS_WEIGHTS[field]
            )

    weights = np.asarray(
        weight_values,
        dtype=float,
    )
    weights /= weights.sum()

    scores = standardised.to_numpy(
        dtype=float
    ) @ weights

    return minmax_scale(
        pd.Series(scores, index=dataframe.index)
    )


def resolve_coordinates(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    x_candidates = (
        "coord_x",
        "x",
        "centroid_x",
        "longitude",
        "lon",
    )
    y_candidates = (
        "coord_y",
        "y",
        "centroid_y",
        "latitude",
        "lat",
    )

    x_field = next(
        (
            field
            for field in x_candidates
            if field in dataframe.columns
        ),
        None,
    )
    y_field = next(
        (
            field
            for field in y_candidates
            if field in dataframe.columns
        ),
        None,
    )

    if x_field is None or y_field is None:
        raise ValueError(
            "The dataset must contain coordinate fields, such as "
            "x/y or longitude/latitude."
        )

    return (
        numeric_series(dataframe, x_field),
        numeric_series(dataframe, y_field),
    )


def prepare_modelling_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    park_field = resolve_park_field(dataframe)
    coordinate_x, coordinate_y = resolve_coordinates(
        dataframe
    )

    attractiveness = calculate_thesis_attractiveness(
        dataframe,
        park_field,
    )
    attractiveness_thesis_scale = (
        attractiveness * 5.0
    )

    features = pd.DataFrame(
        {
            "distancia_media_servicos_log": np.log1p(
                numeric_series(
                    dataframe,
                    "distancia_media_servicos",
                ).clip(lower=0)
            ),
            "Farmacias": numeric_series(
                dataframe,
                "Farmacias",
            ),
            "Supermercados": numeric_series(
                dataframe,
                "Supermercados",
            ),
            "coord_y": coordinate_y,
            "pop_64_mais_log": np.log1p(
                numeric_series(
                    dataframe,
                    "pop_64_mais",
                ).clip(lower=0)
            ),
            "coord_x": coordinate_x,
            park_field: numeric_series(
                dataframe,
                park_field,
            ),
            "Centro Saude": numeric_series(
                dataframe,
                "Centro Saude",
            ),
            "CTT": numeric_series(
                dataframe,
                "CTT",
            ),
            "Hospitais": numeric_series(
                dataframe,
                "Hospitais",
            ),
            "atratividade_escala_tese": (
                attractiveness_thesis_scale
            ),
        },
        index=dataframe.index,
    )

    eligible = (
        attractiveness_thesis_scale
        >= ATTRACTIVENESS_THRESHOLD
    )

    allocation_basis = attractiveness.where(
        eligible,
        0.0,
    )
    allocation_total = float(
        allocation_basis.sum()
    )

    if allocation_total <= 0:
        raise ValueError(
            "No building meets the thesis attractiveness threshold."
        )

    observed_agents = (
        allocation_basis
        * (TOTAL_AGENTS / allocation_total)
    )

    reference = dataframe.copy()
    reference["atratividade_pca"] = attractiveness
    reference["atratividade_escala_tese"] = (
        attractiveness_thesis_scale
    )
    reference["agentes_observados"] = (
        observed_agents
    )

    return features, observed_agents, reference


def fit_thesis_model(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
) -> ModelResults:
    model_features = features.drop(
        columns=["atratividade_escala_tese"]
    )

    x_train, x_test, y_train, y_test = (
        train_test_split(
            model_features,
            target,
            test_size=test_size,
            random_state=RANDOM_STATE,
        )
    )

    model = XGBRegressorNormalized(
        total_agents=TOTAL_AGENTS,
        atratividade_minima=ATTRACTIVENESS_THRESHOLD,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.3,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    raw_predictions = np.asarray(
        model.model.predict(x_test),
        dtype=float,
    )
    predictions = np.clip(
        raw_predictions,
        0.0,
        None,
    )
    residuals = (
        y_test.to_numpy(dtype=float)
        - predictions
    )

    return ModelResults(
        model=model,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        residuals=residuals,
        r2=float(
            r2_score(
                y_test,
                predictions,
            )
        ),
        rmse=float(
            mean_squared_error(
                y_test,
                predictions,
            )
            ** 0.5
        ),
        mae=float(
            mean_absolute_error(
                y_test,
                predictions,
            )
        ),
    )


def calculate_shap_values(
    model: Any,
    sample: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate exact TreeSHAP contributions with the XGBoost booster.

    XGBoost returns one contribution per feature plus a final bias term.
    """
    import xgboost as xgb

    data_matrix = xgb.DMatrix(
        sample,
        feature_names=list(sample.columns),
    )
    contributions = model.get_booster().predict(
        data_matrix,
        pred_contribs=True,
    )

    values = np.asarray(
        contributions[:, :-1],
        dtype=float,
    )
    base_values = np.asarray(
        contributions[:, -1],
        dtype=float,
    )

    return values, base_values


try:
    data = load_data()
except Exception as error:
    st.error(
        f"The research dataset could not be loaded: {error}"
    )
    st.stop()

if not SKLEARN_AVAILABLE:
    st.error(
        "Add `scikit-learn` to requirements.txt."
    )
    st.stop()

if not XGBOOST_AVAILABLE:
    st.error(
        "Add `xgboost` to requirements.txt."
    )
    st.stop()

with st.sidebar:
    st.subheader("SHAP Analysis controls")
    st.caption(f"Build: {BUILD_ID}")

    test_percentage = st.slider(
        "Test sample (%)",
        min_value=15,
        max_value=40,
        value=20,
        step=5,
    )

    shap_sample_size = st.slider(
        "Buildings in the SHAP sample",
        min_value=250,
        max_value=5_000,
        value=1_000,
        step=250,
    )


try:
    features, target, result = (
        prepare_modelling_data(data)
    )
except ValueError as error:
    st.error(str(error))
    st.stop()


with st.spinner(
    "Training the thesis XGBoost model..."
):
    model_results = fit_thesis_model(
        features,
        target,
        test_percentage / 100,
    )


metric_columns = st.columns(4)

metric_columns[0].metric(
    "Buildings",
    f"{len(result):,}",
)
metric_columns[1].metric(
    "R²",
    f"{model_results.r2:.3f}",
)
metric_columns[2].metric(
    "RMSE",
    f"{model_results.rmse:.3f}",
)
metric_columns[3].metric(
    "MAE",
    f"{model_results.mae:.3f}",
)


(
    importance_tab,
    summary_tab,
    predictions_tab,
    residuals_tab,
    methodology_tab,
) = st.tabs(
    [
        "Global importance",
        "SHAP summary",
        "Predictions",
        "Residuals",
        "Methodology",
    ]
)


sample_size = min(
    shap_sample_size,
    len(model_results.x_test),
)

shap_sample = model_results.x_test.sample(
    sample_size,
    random_state=RANDOM_STATE,
)

with st.spinner(
    "Calculating SHAP values..."
):
    shap_values, base_values = (
        calculate_shap_values(
            model_results.model.model,
            shap_sample,
        )
    )


with importance_tab:
    st.subheader(
        "Mean SHAP Importance"
    )

    mean_absolute_shap = np.abs(
        shap_values
    ).mean(axis=0)

    importance = pd.DataFrame(
        {
            "Variable": [
                display_label(field)
                for field in shap_sample.columns
            ],
            "Mean absolute SHAP value": (
                mean_absolute_shap
            ),
        }
    ).sort_values(
        "Mean absolute SHAP value",
        ascending=True,
    )

    importance_chart = px.bar(
        importance,
        x="Mean absolute SHAP value",
        y="Variable",
        orientation="h",
        title=(
            "Mean Variable Importance "
            "Based on SHAP Values"
        ),
    )

    importance_chart.update_layout(
        xaxis_title=(
            "Mean absolute SHAP value"
        ),
        yaxis_title="",
    )

    st.plotly_chart(
        importance_chart,
        width="stretch",
    )

    st.dataframe(
        importance.sort_values(
            "Mean absolute SHAP value",
            ascending=False,
        ).round(4),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Mean absolute SHAP values measure the average magnitude of each "
        "variable's contribution. They do not indicate causality."
    )


with summary_tab:
    st.subheader(
        "SHAP Summary"
    )

    feature_names = [
        display_label(field)
        for field in shap_sample.columns
    ]

    order = np.argsort(
        np.abs(shap_values).mean(axis=0)
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    for display_position, feature_index in enumerate(order):
        values = shap_values[:, feature_index]
        feature_values = shap_sample.iloc[
            :,
            feature_index,
        ].to_numpy(dtype=float)

        spread = float(
            np.nanmax(feature_values)
            - np.nanmin(feature_values)
        )
        if spread > 0:
            normalised_values = (
                feature_values
                - np.nanmin(feature_values)
            ) / spread
        else:
            normalised_values = np.full(
                len(feature_values),
                0.5,
            )

        jitter = rng.normal(
            0.0,
            0.08,
            size=len(values),
        )

        scatter = axis.scatter(
            values,
            display_position + jitter,
            c=normalised_values,
            cmap="coolwarm",
            s=12,
            alpha=0.65,
            linewidths=0,
        )

    axis.axvline(
        0,
        linewidth=1,
    )
    axis.set_yticks(
        range(len(order))
    )
    axis.set_yticklabels(
        [
            feature_names[index]
            for index in order
        ]
    )
    axis.set_xlabel(
        "SHAP value"
    )
    axis.set_ylabel("")
    axis.set_title(
        "SHAP Summary (Beeswarm)"
    )

    colour_bar = figure.colorbar(
        scatter,
        ax=axis,
        pad=0.02,
    )
    colour_bar.set_label(
        "Feature value"
    )
    colour_bar.set_ticks(
        [0, 1]
    )
    colour_bar.set_ticklabels(
        ["Low", "High"]
    )

    figure.tight_layout()

    st.pyplot(
        figure,
        clear_figure=True,
    )

    st.caption(
        "Each point represents one building. Position shows whether a variable "
        "increases or decreases the prediction; colour represents the variable value."
    )


with predictions_tab:
    st.subheader(
        "Observed and Predicted Agents"
    )

    prediction_frame = pd.DataFrame(
        {
            "Observed agents": (
                model_results.y_test
                .to_numpy(dtype=float)
            ),
            "Predicted agents": (
                model_results.predictions
            ),
        }
    )

    prediction_chart = px.scatter(
        prediction_frame,
        x="Observed agents",
        y="Predicted agents",
        opacity=0.55,
        title=(
            "Observed Versus Predicted "
            "Agents per Building"
        ),
    )

    lower_bound = float(
        min(
            prediction_frame.min()
        )
    )
    upper_bound = float(
        max(
            prediction_frame.max()
        )
    )

    prediction_chart.add_shape(
        type="line",
        x0=lower_bound,
        y0=lower_bound,
        x1=upper_bound,
        y1=upper_bound,
        line={
            "dash": "dash",
        },
    )

    prediction_chart.update_layout(
        xaxis_title="Observed agents",
        yaxis_title="Predicted agents",
    )

    st.plotly_chart(
        prediction_chart,
        width="stretch",
    )


with residuals_tab:
    st.subheader(
        "Model Residuals"
    )

    residual_frame = pd.DataFrame(
        {
            "Predicted agents": (
                model_results.predictions
            ),
            "Residual": (
                model_results.residuals
            ),
        }
    )

    residual_columns = st.columns(2)

    with residual_columns[0]:
        residual_distribution = px.histogram(
            residual_frame,
            x="Residual",
            nbins=50,
            title=(
                "Residual Distribution"
            ),
        )

        residual_distribution.update_layout(
            xaxis_title="Residual",
            yaxis_title="Buildings",
            bargap=0.04,
        )

        st.plotly_chart(
            residual_distribution,
            width="stretch",
        )

    with residual_columns[1]:
        residual_scatter = px.scatter(
            residual_frame,
            x="Predicted agents",
            y="Residual",
            opacity=0.5,
            title=(
                "Residuals Versus "
                "Predicted Values"
            ),
        )

        residual_scatter.add_hline(
            y=0,
            line_dash="dash",
        )

        residual_scatter.update_layout(
            xaxis_title="Predicted agents",
            yaxis_title="Residual",
        )

        st.plotly_chart(
            residual_scatter,
            width="stretch",
        )

    st.info(
        "Spatial autocorrelation and LISA analysis of the standardised residuals "
        "are presented in the Spatial Diagnostics module to avoid duplication."
    )


with methodology_tab:
    st.subheader(
        "Thesis Workflow"
    )

    st.markdown(
        """
This module follows **Code 12 of the doctoral thesis**:

1. The urban-attractiveness score is reconstructed from the nine socio-spatial
   indicators used in the PCA workflow.
2. The score is converted to the thesis scale and a minimum attractiveness
   threshold of **2.6** is applied.
3. A total of **22,000 agents** is allocated proportionally across eligible
   buildings.
4. The explanatory variables follow Figures 61 and 62: mean distance to
   services (log), pharmacies, supermarkets, coordinates Y and X, population
   aged 65 or over (log), parks and gardens, health centres, post offices and
   hospitals.
5. The model is an XGBoost regressor with **100 estimators**, maximum depth
   **3**, learning rate **0.3** and random state **42**.
6. Global importance is calculated from the mean absolute SHAP values.
7. The beeswarm shows the direction and magnitude of each variable's
   contribution for individual buildings.
8. Residual distribution and residuals versus predicted values reproduce the
   non-spatial diagnostics associated with the model.

### Interpretation limits

SHAP explains how the fitted XGBoost model uses the available variables. It
does not establish causal relationships. The PCA construction of the
attractiveness index, the interpretation of K-means clusters and LIME
explanations are presented in their dedicated modules.
        """
    )

    variable_table = pd.DataFrame(
        {
            "Dataset field": (
                model_results.x_train.columns
            ),
            "Variable": [
                display_label(field)
                for field
                in model_results.x_train.columns
            ],
        }
    )

    st.dataframe(
        variable_table,
        width="stretch",
        hide_index=True,
    )


download = result.copy()

full_features = features.drop(
    columns=["atratividade_escala_tese"]
)

download["agentes_estimados_xgboost"] = (
    np.clip(
        model_results.model.model.predict(
            full_features
        ),
        0.0,
        None,
    )
)

download_columns = [
    field
    for field in [
        "osm_id",
        "designacao_simplificada",
        "atratividade_pca",
        "atratividade_escala_tese",
        "agentes_observados",
        "agentes_estimados_xgboost",
        "latitude",
        "longitude",
    ]
    if field in download.columns
]

st.download_button(
    "Download SHAP Analysis results",
    data=download[
        download_columns
    ].to_csv(
        index=False
    ).encode(
        "utf-8"
    ),
    file_name=(
        "geoinsightlab_shap_analysis.csv"
    ),
    mime="text/csv",
)
