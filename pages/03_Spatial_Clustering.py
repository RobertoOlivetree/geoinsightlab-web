import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import column_label, load_data
from utils.maps import render_categorical_polygon_map
from utils.theme import apply_theme, page_header, scientific_note


apply_theme()
page_header(
    "Spatial Clustering",
    "Explore the spatial patterns and urban profiles identified by three clustering algorithms.",
)

scientific_note(
    "This module visualises clustering results already stored in the research dataset. "
    "It does not retrain the algorithms during each session, which preserves reproducibility "
    "and ensures efficient performance on Streamlit Community Cloud."
)

CLUSTER_METHODS = {
    "K-Means": "cluster_kmeans",
    "Gaussian Mixture Model": "cluster_gmm",
    "Agglomerative Clustering": "cluster_agglo",
}

PROFILE_FEATURES = [
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
]

PCA_FEATURES = [
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
]

CLUSTER_COLOURS = {
    "Cluster 0": [46, 139, 87, 190],
    "Cluster 1": [255, 165, 0, 190],
    "Cluster 2": [65, 105, 225, 190],
    "Cluster 3": [186, 85, 211, 190],
}


@st.cache_data(show_spinner=False)
def compute_pca_projection(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    working = dataframe[PCA_FEATURES].apply(pd.to_numeric, errors="coerce").copy()
    working = working.fillna(working.median(numeric_only=True))

    standard_deviation = working.std(ddof=0).replace(0, 1)
    standardised = (working - working.mean()) / standard_deviation

    _, singular_values, vectors_transposed = np.linalg.svd(
        standardised.to_numpy(dtype=float),
        full_matrices=False,
    )
    scores = standardised.to_numpy(dtype=float) @ vectors_transposed.T[:, :2]
    explained = (singular_values**2) / np.sum(singular_values**2)

    result = dataframe[
        [
            "osm_id",
            "designacao_simplificada",
            "cluster_kmeans",
            "cluster_gmm",
            "cluster_agglo",
        ]
    ].copy()
    result["PC1"] = scores[:, 0]
    result["PC2"] = scores[:, 1]
    return result, explained[:2]


@st.cache_data(show_spinner=False)
def cluster_summary(dataframe: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    summary = (
        dataframe.groupby(cluster_column, dropna=False)
        .agg(
            Buildings=("osm_id", "count"),
            Parishes=("designacao_simplificada", "nunique"),
            **{
                "Mean nearby services": ("numero_servicos_proximos", "mean"),
                "Mean distance to services (m)": (
                    "distancia_media_servicos",
                    "mean",
                ),
                "Mean population aged 65+": ("pop_64_mais", "mean"),
            },
        )
        .reset_index()
        .rename(columns={cluster_column: "Cluster"})
    )
    summary["Cluster"] = summary["Cluster"].map(lambda value: f"Cluster {int(value)}")
    summary["Share (%)"] = 100 * summary["Buildings"] / summary["Buildings"].sum()
    return summary


def render_cluster_map(
    dataframe: pd.DataFrame,
    cluster_column: str,
    maximum_polygons: int,
) -> None:
    render_categorical_polygon_map(
        dataframe=dataframe,
        category_column=cluster_column,
        colour_map=CLUSTER_COLOURS,
        maximum_polygons=maximum_polygons,
        category_prefix="Cluster",
    )

def adjusted_agreement(first: pd.Series, second: pd.Series) -> float:
    comparison = pd.crosstab(first, second)
    if comparison.empty:
        return float("nan")
    direct = np.trace(comparison.to_numpy())
    reversed_match = np.trace(np.fliplr(comparison.to_numpy()))
    return max(direct, reversed_match) / comparison.to_numpy().sum()


try:
    data = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.subheader("Clustering controls")
    selected_method = st.selectbox("Method", list(CLUSTER_METHODS))
    cluster_column = CLUSTER_METHODS[selected_method]

    parishes = sorted(data["designacao_simplificada"].dropna().unique().tolist())
    selected_parishes = st.multiselect(
        "Parishes",
        options=parishes,
        default=parishes,
    )
    maximum_points = st.slider(
        "Maximum building polygons on the map",
        min_value=1_000,
        max_value=15_000,
        value=6_000,
        step=1_000,
    )

filtered = data[data["designacao_simplificada"].isin(selected_parishes)].copy()
if filtered.empty:
    st.warning("Select at least one parish to display the clustering results.")
    st.stop()

summary = cluster_summary(filtered, cluster_column)
cluster_counts = filtered[cluster_column].value_counts().sort_index()

metric_columns = st.columns(4)
metric_columns[0].metric("Buildings", f"{len(filtered):,}")
metric_columns[1].metric("Clusters", f"{filtered[cluster_column].nunique()}")
metric_columns[2].metric(
    "Largest cluster",
    f"{int(cluster_counts.max()):,}",
)
metric_columns[3].metric(
    "Smallest cluster",
    f"{int(cluster_counts.min()):,}",
)

map_tab, profiles_tab, pca_tab, comparison_tab, methodology_tab = st.tabs(
    [
        "Cluster map",
        "Cluster profiles",
        "PCA projection",
        "Method comparison",
        "Methodology",
    ]
)

with map_tab:
    st.subheader(f"{selected_method} spatial distribution")
    st.caption(
        "Clusters are represented through the real Polygon or MultiPolygon building "
        "footprints stored in geometry_wkt. Artificial point or geometric proxies are not used."
    )
    render_cluster_map(filtered, cluster_column, maximum_points)

    distribution = summary[["Cluster", "Buildings", "Share (%)"]].copy()
    figure = px.bar(
        distribution,
        x="Cluster",
        y="Buildings",
        text="Share (%)",
        title="Cluster size",
    )
    figure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    figure.update_layout(yaxis_title="Number of buildings", xaxis_title="")
    st.plotly_chart(figure, use_container_width=True)

with profiles_tab:
    st.subheader("Descriptive cluster profiles")
    st.caption(
        "Profiles are based on mean values within each cluster. They describe the "
        "stored solution and should not be interpreted as causal relationships."
    )

    profile_table = (
        filtered.groupby(cluster_column)[PROFILE_FEATURES]
        .mean(numeric_only=True)
        .rename_axis("Cluster")
        .reset_index()
    )
    profile_table["Cluster"] = profile_table["Cluster"].map(
        lambda value: f"Cluster {int(value)}"
    )

    display_table = profile_table.rename(
        columns={feature: column_label(feature) for feature in PROFILE_FEATURES}
    )
    numeric_display = display_table.select_dtypes(include="number").columns
    display_table[numeric_display] = display_table[numeric_display].round(2)
    st.dataframe(display_table, use_container_width=True, hide_index=True)

    selected_profile_features = st.multiselect(
        "Variables shown in the profile chart",
        options=PROFILE_FEATURES,
        default=[
            "numero_servicos_proximos",
            "distancia_media_servicos",
            "pop_64_mais",
            "Farmacias",
            "Supermercados",
        ],
        format_func=column_label,
    )

    if selected_profile_features:
        chart_data = profile_table.set_index("Cluster")[selected_profile_features]
        means = chart_data.mean(axis=0)
        standard_deviations = chart_data.std(axis=0, ddof=0).replace(0, 1)
        standardised_profiles = (chart_data - means) / standard_deviations
        long_profiles = (
            standardised_profiles.reset_index()
            .melt(id_vars="Cluster", var_name="Indicator", value_name="Standardised mean")
        )
        long_profiles["Indicator"] = long_profiles["Indicator"].map(column_label)
        figure = px.line(
            long_profiles,
            x="Indicator",
            y="Standardised mean",
            color="Cluster",
            markers=True,
            title="Standardised cluster profiles",
        )
        figure.update_layout(xaxis_title="", legend_title="")
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Cluster composition by parish")
    parish_composition = pd.crosstab(
        filtered["designacao_simplificada"],
        filtered[cluster_column],
        normalize="index",
    )
    parish_composition.columns = [f"Cluster {int(value)}" for value in parish_composition.columns]
    parish_composition = (100 * parish_composition).reset_index()
    parish_long = parish_composition.melt(
        id_vars="designacao_simplificada",
        var_name="Cluster",
        value_name="Share (%)",
    )
    parish_figure = px.bar(
        parish_long,
        x="designacao_simplificada",
        y="Share (%)",
        color="Cluster",
        title="Cluster composition within each parish",
    )
    parish_figure.update_layout(
        xaxis_title="Parish",
        yaxis_title="Share of buildings (%)",
        barmode="stack",
    )
    st.plotly_chart(parish_figure, use_container_width=True)

with pca_tab:
    st.subheader("Two-dimensional PCA projection")
    pca_data, explained_variance = compute_pca_projection(filtered)
    pca_data["Cluster"] = pca_data[cluster_column].map(
        lambda value: f"Cluster {int(value)}"
    )

    sample_size = min(8_000, len(pca_data))
    if len(pca_data) > sample_size:
        pca_plot = pca_data.sample(sample_size, random_state=42)
    else:
        pca_plot = pca_data

    figure = px.scatter(
        pca_plot,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=["osm_id", "designacao_simplificada"],
        opacity=0.58,
        title="PCA projection of demographic and accessibility indicators",
    )
    figure.update_layout(
        xaxis_title=f"PC1 ({explained_variance[0] * 100:.1f}% explained variance)",
        yaxis_title=f"PC2 ({explained_variance[1] * 100:.1f}% explained variance)",
        legend_title="",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "PCA is calculated within the application from standardised indicators solely "
        "for visual exploration. The stored cluster assignments remain unchanged."
    )

with comparison_tab:
    st.subheader("Agreement between clustering methods")
    method_columns = list(CLUSTER_METHODS.items())
    agreement_matrix = pd.DataFrame(
        index=[name for name, _ in method_columns],
        columns=[name for name, _ in method_columns],
        dtype=float,
    )

    for first_name, first_column in method_columns:
        for second_name, second_column in method_columns:
            agreement_matrix.loc[first_name, second_name] = adjusted_agreement(
                filtered[first_column],
                filtered[second_column],
            )

    heatmap = go.Figure(
        data=go.Heatmap(
            z=agreement_matrix.to_numpy(),
            x=agreement_matrix.columns,
            y=agreement_matrix.index,
            zmin=0,
            zmax=1,
            text=np.round(100 * agreement_matrix.to_numpy(), 1),
            texttemplate="%{text}%",
            colorbar={"title": "Agreement"},
        )
    )
    heatmap.update_layout(title="Label-adjusted assignment agreement")
    st.plotly_chart(heatmap, use_container_width=True)

    st.dataframe(
        (100 * agreement_matrix).round(1).astype(str) + "%",
        use_container_width=True,
    )
    st.caption(
        "Because cluster labels are arbitrary, the agreement calculation accepts the "
        "direct or reversed two-cluster labelling, whichever produces the stronger match."
    )

    st.subheader("Cross-tabulation")
    first_method = st.selectbox(
        "First method",
        list(CLUSTER_METHODS),
        index=0,
        key="first_comparison_method",
    )
    second_method = st.selectbox(
        "Second method",
        list(CLUSTER_METHODS),
        index=1,
        key="second_comparison_method",
    )
    cross_table = pd.crosstab(
        filtered[CLUSTER_METHODS[first_method]],
        filtered[CLUSTER_METHODS[second_method]],
        margins=True,
    )
    cross_table.index = [
        "Total" if value == "All" else f"Cluster {value}" for value in cross_table.index
    ]
    cross_table.columns = [
        "Total" if value == "All" else f"Cluster {value}" for value in cross_table.columns
    ]
    st.dataframe(cross_table, use_container_width=True)

with methodology_tab:
    st.subheader("Analytical scope")
    st.markdown(
        """
The dataset contains three previously computed cluster assignments for each building:

- **K-Means**, a centroid-based partitioning method;
- **Gaussian Mixture Model**, a probabilistic model based on Gaussian components;
- **Agglomerative Clustering**, a hierarchical bottom-up method.

This page compares the stored assignments using demographic, service-accessibility and
spatial variables. The interactive PCA projection is recalculated from standardised
indicators to provide a common two-dimensional visual reference.

### Interpretation limits

Cluster identifiers are nominal labels rather than ordinal scores. A building in
Cluster 1 is not automatically better or worse than a building in Cluster 0. Meaning
must be derived from the indicator profiles shown in this module and from the validated
interpretation reported in the underlying research.
        """
    )

    st.subheader("Indicators used for PCA and profiling")
    indicator_table = pd.DataFrame(
        {
            "Dataset field": PCA_FEATURES,
            "Display label": [column_label(feature) for feature in PCA_FEATURES],
        }
    )
    st.dataframe(indicator_table, use_container_width=True, hide_index=True)

    st.download_button(
        "Download selected clustering data",
        data=filtered[
            [
                "osm_id",
                "designacao_simplificada",
                *PCA_FEATURES,
                "cluster_kmeans",
                "cluster_gmm",
                "cluster_agglo",
                "latitude",
                "longitude",
            ]
        ].to_csv(index=False).encode("utf-8"),
        file_name="geoinsightlab_spatial_clustering.csv",
        mime="text/csv",
    )
