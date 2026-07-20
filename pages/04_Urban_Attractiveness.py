import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import column_label, load_data
from utils.maps import render_continuous_polygon_map
from utils.theme import apply_theme, page_header, scientific_note


apply_theme()

page_header(
    "Urban Attractiveness",
    "Explore the building-level urban attractiveness index developed in the thesis.",
)

scientific_note(
    "The Urban Attractiveness Index combines multiple accessibility indicators "
    "using Principal Component Analysis (PCA) to characterise the spatial "
    "distribution of urban attractiveness across Porto."
)


FIELD_ALIASES = {
    "parks": (
        "Parques ou jardins",
        "Parques e jardins",
    ),
}

BASE_FEATURES = [
    "Bancos",
    "Centro Saude",
    "Farmacias",
    "Supermercados",
    "Hospitais",
    "CTT",
    "pop_64_mais",
    "distancia_media_servicos",
]

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

DISPLAY_LABELS = {
    "Bancos": "Banks",
    "Centro Saude": "Health centres",
    "Farmacias": "Pharmacies",
    "Supermercados": "Supermarkets",
    "Hospitais": "Hospitals",
    "CTT": "Post offices",
    "pop_64_mais": "Population aged 65+",
    "distancia_media_servicos": "Mean distance to services",
    "prox_servicos": "Proximity to services",
    "atratividade_pca": "PCA-derived attractiveness",
    "atratividade_equal": "Equal-weight attractiveness",
    "atratividade_tese": "Thesis-weight attractiveness",
}


def display_label(field: str) -> str:
    return DISPLAY_LABELS.get(
        field,
        column_label(field),
    )


def resolve_park_field(
    dataframe: pd.DataFrame,
) -> str:
    for candidate in FIELD_ALIASES["parks"]:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        "The dataset must contain either "
        "'Parques ou jardins' or 'Parques e jardins'."
    )


def winsorise(
    series: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    lower = numeric.quantile(0.01)
    upper = numeric.quantile(0.99)

    return numeric.clip(
        lower=lower,
        upper=upper,
    )


def minmax_scale(
    values: np.ndarray,
) -> np.ndarray:
    minimum = float(
        np.nanmin(values)
    )
    maximum = float(
        np.nanmax(values)
    )

    value_range = maximum - minimum

    if value_range <= 0:
        return np.zeros(
            len(values),
            dtype=float,
        )

    return (
        values - minimum
    ) / value_range


@st.cache_data(show_spinner=False)
def calculate_indices(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    float,
    str,
]:
    park_field = resolve_park_field(
        dataframe
    )

    features = [
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
        set(features).difference(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            "The Urban Attractiveness Index cannot "
            "be calculated because the following "
            "fields are missing: "
            + ", ".join(missing)
        )

    index_data = dataframe[
        features
    ].copy()

    index_data = index_data.apply(
        winsorise
    )

    index_data["prox_servicos"] = (
        -index_data.pop(
            "distancia_media_servicos"
        )
    )

    means = index_data.mean(
        axis=0
    )

    standard_deviations = (
        index_data.std(
            axis=0,
            ddof=0,
        ).replace(
            0,
            1,
        )
    )

    standardised = (
        index_data - means
    ) / standard_deviations

    matrix = standardised.to_numpy(
        dtype=float
    )

    (
        _,
        singular_values,
        vectors_transposed,
    ) = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    loadings = (
        vectors_transposed[0].copy()
    )

    scores = matrix @ loadings

    supermarket_position = list(
        index_data.columns
    ).index(
        "Supermercados"
    )

    supermarket_correlation = (
        np.corrcoef(
            matrix[
                :,
                supermarket_position,
            ],
            scores,
        )[0, 1]
    )

    if supermarket_correlation < 0:
        loadings *= -1
        scores *= -1

    pca_index = minmax_scale(
        scores
    )

    equal_scores = (
        standardised.mean(
            axis=1
        ).to_numpy(
            dtype=float
        )
    )

    equal_index = minmax_scale(
        equal_scores
    )

    thesis_weight_vector = []

    for field in index_data.columns:
        if field == park_field:
            thesis_weight_vector.append(
                THESIS_WEIGHTS["parks"]
            )
        else:
            thesis_weight_vector.append(
                THESIS_WEIGHTS[field]
            )

    thesis_weight_vector = np.asarray(
        thesis_weight_vector,
        dtype=float,
    )

    thesis_weight_vector /= (
        thesis_weight_vector.sum()
    )

    thesis_scores = (
        matrix
        @ thesis_weight_vector
    )

    thesis_index = minmax_scale(
        thesis_scores
    )

    result = dataframe.copy()

    result["atratividade_pca"] = (
        pca_index
    )

    result["atratividade_equal"] = (
        equal_index
    )

    result["atratividade_tese"] = (
        thesis_index
    )

    absolute_loadings = np.abs(
        loadings
    )

    loading_total = (
        absolute_loadings.sum()
    )

    if loading_total <= 0:
        pca_weights = np.repeat(
            1 / len(
                absolute_loadings
            ),
            len(
                absolute_loadings
            ),
        )
    else:
        pca_weights = (
            absolute_loadings
            / loading_total
        )

    weights_table = pd.DataFrame(
        {
            "Dataset field": (
                index_data.columns
            ),
            "PCA loading": loadings,
            "PCA weight": pca_weights,
            "Thesis weight": (
                thesis_weight_vector
            ),
        }
    )

    weights_table["Indicator"] = (
        weights_table[
            "Dataset field"
        ]
        .replace(
            {
                park_field: (
                    "Parks and gardens"
                ),
            }
        )
        .map(
            lambda value: (
                display_label(value)
                if value
                != "Parks and gardens"
                else value
            )
        )
    )

    weights_table[
        "PCA weight (%)"
    ] = (
        100
        * weights_table[
            "PCA weight"
        ]
    )

    weights_table[
        "Thesis weight (%)"
    ] = (
        100
        * weights_table[
            "Thesis weight"
        ]
    )

    explained_variance = (
        singular_values[0] ** 2
        / np.sum(
            singular_values**2
        )
    )

    return (
        result,
        weights_table,
        float(
            explained_variance
        ),
        park_field,
    )


@st.cache_data(show_spinner=False)
def parish_summary(
    dataframe: pd.DataFrame,
    selected_index: str,
) -> pd.DataFrame:
    summary = (
        dataframe.groupby(
            "designacao_simplificada",
            dropna=False,
        )
        .agg(
            Buildings=(
                "osm_id",
                "count",
            ),
            Mean_attractiveness=(
                selected_index,
                "mean",
            ),
            Median_attractiveness=(
                selected_index,
                "median",
            ),
            Mean_nearby_services=(
                "numero_servicos_proximos",
                "mean",
            ),
            Mean_distance_to_services=(
                "distancia_media_servicos",
                "mean",
            ),
            Population_65_plus=(
                "pop_64_mais",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "designacao_simplificada": (
                    "Parish"
                ),
                "Mean_attractiveness": (
                    "Mean attractiveness"
                ),
                "Median_attractiveness": (
                    "Median attractiveness"
                ),
                "Mean_nearby_services": (
                    "Mean nearby services"
                ),
                "Mean_distance_to_services": (
                    "Mean distance to services (m)"
                ),
                "Population_65_plus": (
                    "Population aged 65+"
                ),
            }
        )
        .sort_values(
            "Mean attractiveness",
            ascending=False,
        )
    )

    return summary


try:
    data = load_data()

    (
        data,
        weights_table,
        explained_variance,
        park_field,
    ) = calculate_indices(
        data
    )

except (
    FileNotFoundError,
    ValueError,
) as error:
    st.error(
        str(error)
    )
    st.stop()


with st.sidebar:
    st.subheader(
        "Attractiveness controls"
    )

    parishes = sorted(
        data[
            "designacao_simplificada"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    selected_parishes = st.multiselect(
        "Parishes",
        options=parishes,
        default=parishes,
    )

    formulation = st.selectbox(
        "Index formulation",
        [
            "PCA-derived weights",
            "Thesis-reported weights",
            "Equal weights",
        ],
    )

    maximum_polygons = st.slider(
        "Maximum building polygons on the map",
        min_value=1_000,
        max_value=15_000,
        value=6_000,
        step=1_000,
    )


INDEX_COLUMNS = {
    "PCA-derived weights": (
        "atratividade_pca"
    ),
    "Thesis-reported weights": (
        "atratividade_tese"
    ),
    "Equal weights": (
        "atratividade_equal"
    ),
}

selected_index = (
    INDEX_COLUMNS[
        formulation
    ]
)

filtered = data[
    data[
        "designacao_simplificada"
    ].isin(
        selected_parishes
    )
].copy()


if filtered.empty:
    st.warning(
        "Select at least one parish to display "
        "the attractiveness results."
    )
    st.stop()


metric_columns = st.columns(
    4
)

metric_columns[0].metric(
    "Buildings",
    f"{len(filtered):,}",
)

metric_columns[1].metric(
    "Mean attractiveness",
    f"{filtered[selected_index].mean():.3f}",
)

metric_columns[2].metric(
    "Maximum attractiveness",
    f"{filtered[selected_index].max():.3f}",
)

metric_columns[3].metric(
    "PC1 variance explained",
    f"{100 * explained_variance:.1f}%",
)


(
    map_tab,
    rankings_tab,
    weights_tab,
    comparison_tab,
    methodology_tab,
) = st.tabs(
    [
        "Attractiveness map",
        "Parish rankings",
        "PCA weights",
        "Index comparison",
        "Methodology",
    ]
)


with map_tab:
    st.subheader(
        "Urban Attractiveness Map"
    )

    render_continuous_polygon_map(
        dataframe=filtered,
        value_column=selected_index,
        maximum_polygons=maximum_polygons,
    )

    distribution = px.histogram(
        filtered,
        x=selected_index,
        nbins=40,
        title=(
            "Distribution of building-level "
            "attractiveness"
        ),
        labels={
            selected_index: (
                display_label(
                    selected_index
                )
            ),
        },
    )

    distribution.update_layout(
        xaxis_title=display_label(
            selected_index
        ),
        yaxis_title=(
            "Number of buildings"
        ),
        bargap=0.04,
    )

    st.plotly_chart(
        distribution,
        width="stretch",
    )


with rankings_tab:
    st.subheader(
        "Attractiveness by parish"
    )

    summary = parish_summary(
        filtered,
        selected_index,
    )

    chart = px.bar(
        summary.sort_values(
            "Mean attractiveness",
            ascending=True,
        ),
        x="Mean attractiveness",
        y="Parish",
        orientation="h",
        title=(
            f"Mean {formulation.lower()} "
            "attractiveness by parish"
        ),
        hover_data=[
            "Buildings",
            "Median attractiveness",
            "Mean nearby services",
            "Mean distance to services (m)",
        ],
    )

    chart.update_layout(
        xaxis_title=(
            "Mean attractiveness"
        ),
        yaxis_title="",
    )

    st.plotly_chart(
        chart,
        width="stretch",
    )

    display_summary = (
        summary.copy()
    )

    numeric_columns = (
        display_summary.select_dtypes(
            include="number"
        ).columns
    )

    display_summary[
        numeric_columns
    ] = (
        display_summary[
            numeric_columns
        ].round(3)
    )

    st.dataframe(
        display_summary,
        width="stretch",
        hide_index=True,
    )


with weights_tab:
    st.subheader(
        "Indicator weights"
    )

    chart_data = (
        weights_table.sort_values(
            "PCA weight (%)",
            ascending=True,
        )
    )

    weights_chart = px.bar(
        chart_data,
        x=[
            "PCA weight (%)",
            "Thesis weight (%)",
        ],
        y="Indicator",
        orientation="h",
        barmode="group",
        title=(
            "PCA-derived weights and "
            "thesis-reported weights"
        ),
    )

    weights_chart.update_layout(
        xaxis_title="Weight (%)",
        yaxis_title="",
        legend_title="",
    )

    st.plotly_chart(
        weights_chart,
        width="stretch",
    )

    display_weights = (
        weights_table[
            [
                "Indicator",
                "PCA loading",
                "PCA weight (%)",
                "Thesis weight (%)",
            ]
        ].copy()
    )

    display_weights[
        "PCA loading"
    ] = (
        display_weights[
            "PCA loading"
        ].round(4)
    )

    display_weights[
        "PCA weight (%)"
    ] = (
        display_weights[
            "PCA weight (%)"
        ].round(2)
    )

    display_weights[
        "Thesis weight (%)"
    ] = (
        display_weights[
            "Thesis weight (%)"
        ].round(2)
    )

    st.dataframe(
        display_weights,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "PCA weights are recalculated from the current "
        "research dataset. Thesis weights reproduce "
        "Table 33 of the submitted thesis."
    )


with comparison_tab:
    st.subheader(
        "Comparison of index formulations"
    )

    sample_size = min(
        8_000,
        len(filtered),
    )

    comparison_sample = (
        filtered.sample(
            sample_size,
            random_state=42,
        )
    )

    comparison = px.scatter(
        comparison_sample,
        x="atratividade_equal",
        y="atratividade_pca",
        color=(
            "designacao_simplificada"
        ),
        opacity=0.55,
        hover_data=[
            "osm_id",
            "numero_servicos_proximos",
            "distancia_media_servicos",
        ],
        title=(
            "PCA-derived and equal-weight "
            "attractiveness"
        ),
        labels={
            "atratividade_equal": (
                "Equal-weight attractiveness"
            ),
            "atratividade_pca": (
                "PCA-derived attractiveness"
            ),
            "designacao_simplificada": (
                "Parish"
            ),
        },
    )

    comparison.update_layout(
        legend_title="Parish",
    )

    st.plotly_chart(
        comparison,
        width="stretch",
    )

    correlations = (
        filtered[
            [
                "atratividade_pca",
                "atratividade_tese",
                "atratividade_equal",
            ]
        ].corr()
    )

    correlation_display = (
        correlations.rename(
            index={
                "atratividade_pca": (
                    "PCA-derived"
                ),
                "atratividade_tese": (
                    "Thesis weights"
                ),
                "atratividade_equal": (
                    "Equal weights"
                ),
            },
            columns={
                "atratividade_pca": (
                    "PCA-derived"
                ),
                "atratividade_tese": (
                    "Thesis weights"
                ),
                "atratividade_equal": (
                    "Equal weights"
                ),
            },
        )
    )

    st.subheader(
        "Pearson correlations"
    )

    st.dataframe(
        correlation_display.round(
            3
        ),
        width="stretch",
    )

    differences = pd.DataFrame(
        {
            "PCA minus equal weights": (
                filtered[
                    "atratividade_pca"
                ]
                - filtered[
                    "atratividade_equal"
                ]
            ),
            "PCA minus thesis weights": (
                filtered[
                    "atratividade_pca"
                ]
                - filtered[
                    "atratividade_tese"
                ]
            ),
        }
    )

    difference_long = (
        differences.melt(
            var_name="Comparison",
            value_name="Difference",
        )
    )

    difference_chart = px.histogram(
        difference_long,
        x="Difference",
        color="Comparison",
        nbins=40,
        barmode="overlay",
        opacity=0.65,
        title=(
            "Differences between index formulations"
        ),
    )

    difference_chart.update_layout(
        xaxis_title=(
            "Difference in attractiveness score"
        ),
        yaxis_title=(
            "Number of buildings"
        ),
        legend_title="",
        bargap=0.04,
    )

    st.plotly_chart(
        difference_chart,
        width="stretch",
    )


with methodology_tab:
    st.subheader(
        "Analytical workflow"
    )

    st.markdown(
        f"""
The module follows the procedure used in **Code 11 of the thesis**:

1. Nine socio-spatial variables are selected: banks, health centres,
   pharmacies, supermarkets, parks and gardens, hospitals, post offices,
   population aged 65 or over, and mean distance to services.
2. Missing values are replaced with zero and every variable is
   **winsorised at the 1st and 99th percentiles**.
3. Mean distance is multiplied by −1 to represent **proximity to services**.
4. All variables are transformed through **z-score standardisation**.
5. A one-component PCA is estimated and its direction is oriented to maintain
   a positive correlation with supermarkets.
6. The first-component scores are rescaled to the interval **[0, 1]**.
7. Two sensitivity formulations are also provided: the weights reported in
   Table 33 of the thesis and an equal-weight index.

The dataset field used for parks and gardens is `{park_field}`.

### Interpretation limits

The index is a relative composite measure within the analysed dataset.
Higher values indicate combinations of service availability, demographic
demand and proximity associated with greater urban attractiveness. The index
does not represent property value, individual preference or causal effects.
        """
    )

    indicator_table = (
        weights_table[
            [
                "Dataset field",
                "Indicator",
            ]
        ].copy()
    )

    st.dataframe(
        indicator_table,
        width="stretch",
        hide_index=True,
    )

    download_columns = [
        "osm_id",
        "designacao_simplificada",
        "Bancos",
        "Centro Saude",
        "Farmacias",
        "Supermercados",
        park_field,
        "Hospitais",
        "CTT",
        "pop_64_mais",
        "distancia_media_servicos",
        "atratividade_pca",
        "atratividade_tese",
        "atratividade_equal",
        "latitude",
        "longitude",
    ]

    st.download_button(
        "Download selected attractiveness data",
        data=filtered[
            download_columns
        ].to_csv(
            index=False
        ).encode(
            "utf-8"
        ),
        file_name=(
            "geoinsightlab_urban_attractiveness.csv"
        ),
        mime="text/csv",
    )
