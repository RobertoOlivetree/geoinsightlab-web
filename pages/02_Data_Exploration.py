import plotly.express as px
import streamlit as st

from utils.data_loader import (
    column_label,
    filter_by_parish,
    load_data,
    numeric_columns,
)
from utils.maps import render_point_map
from utils.theme import apply_theme, page_header, scientific_note


apply_theme()
page_header(
    "Data Exploration",
    "Interactive statistical and spatial exploration of the Porto building-level dataset.",
)

dataframe = load_data()
parish_options = sorted(
    dataframe["designacao_simplificada"].dropna().astype(str).unique().tolist()
)
number_columns = numeric_columns(dataframe)
preferred_variable = "numero_servicos_proximos"
default_variable_index = (
    number_columns.index(preferred_variable) if preferred_variable in number_columns else 0
)

with st.sidebar:
    st.header("Filters")
    selected_parishes = st.multiselect(
        "Parishes",
        options=parish_options,
        default=parish_options,
    )
    selected_variable = st.selectbox(
        "Variable",
        options=number_columns,
        index=default_variable_index,
        format_func=column_label,
    )
    maximum_points = st.slider(
        "Maximum number of map points",
        min_value=2_000,
        max_value=20_000,
        value=10_000,
        step=1_000,
    )

filtered = filter_by_parish(dataframe, selected_parishes)

scientific_note(
    "The dataset contains 31,873 building records and 27 variables, including "
    "demographic attributes, service-access indicators, clustering outputs and geographic coordinates."
)

if filtered.empty:
    st.warning("Select at least one parish to display the results.")
    st.stop()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Buildings", f"{len(filtered):,}")
metric_2.metric("Parishes", filtered["designacao_simplificada"].nunique())
metric_3.metric(
    "Mean nearby services",
    f"{filtered['numero_servicos_proximos'].mean():.2f}",
)
metric_4.metric(
    "Mean distance to services",
    f"{filtered['distancia_media_servicos'].mean():.1f} m",
)

map_tab, distribution_tab, relationships_tab, table_tab = st.tabs(
    ["Spatial distribution", "Distributions", "Relationships", "Data table"]
)

with map_tab:
    st.subheader("Interactive point map")
    st.caption(
        "Values are displayed using a robust colour scale based on the 2nd and 98th percentiles."
    )
    render_point_map(
        filtered,
        value_column=selected_variable,
        maximum_points=maximum_points,
    )

with distribution_tab:
    chart_column_1, chart_column_2 = st.columns(2)
    variable_label = column_label(selected_variable)

    with chart_column_1:
        distribution_figure = px.histogram(
            filtered,
            x=selected_variable,
            nbins=40,
            marginal="box",
            title=f"Distribution of {variable_label}",
            labels={selected_variable: variable_label},
        )
        st.plotly_chart(distribution_figure, use_container_width=True)

    with chart_column_2:
        parish_summary = (
            filtered.groupby("designacao_simplificada", as_index=False)[selected_variable]
            .mean()
            .sort_values(selected_variable, ascending=True)
        )
        parish_figure = px.bar(
            parish_summary,
            x=selected_variable,
            y="designacao_simplificada",
            orientation="h",
            title=f"Mean {variable_label} by parish",
            labels={
                selected_variable: variable_label,
                "designacao_simplificada": "Parish",
            },
        )
        st.plotly_chart(parish_figure, use_container_width=True)

    st.subheader("Descriptive statistics")
    descriptive_statistics = filtered[number_columns].describe().transpose().round(3)
    descriptive_statistics.index = [column_label(column) for column in descriptive_statistics.index]
    st.dataframe(descriptive_statistics, use_container_width=True)

with relationships_tab:
    st.subheader("Variable relationships")
    second_variable = st.selectbox(
        "Comparison variable",
        options=number_columns,
        index=(
            number_columns.index("distancia_media_servicos")
            if "distancia_media_servicos" in number_columns
            else min(1, len(number_columns) - 1)
        ),
        format_func=column_label,
    )

    relationship_column_1, relationship_column_2 = st.columns(2)
    with relationship_column_1:
        scatter_figure = px.scatter(
            filtered,
            x=selected_variable,
            y=second_variable,
            color="designacao_simplificada",
            opacity=0.45,
            render_mode="webgl",
            title=f"{column_label(selected_variable)} vs {column_label(second_variable)}",
            labels={
                selected_variable: column_label(selected_variable),
                second_variable: column_label(second_variable),
                "designacao_simplificada": "Parish",
            },
        )
        st.plotly_chart(scatter_figure, use_container_width=True)

    with relationship_column_2:
        correlation_columns = [
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
        ]
        correlation_columns = [
            column for column in correlation_columns if column in filtered.columns
        ]
        correlation = filtered[correlation_columns].corr(numeric_only=True)
        correlation_figure = px.imshow(
            correlation,
            x=[column_label(column) for column in correlation.columns],
            y=[column_label(column) for column in correlation.index],
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Correlation matrix",
        )
        st.plotly_chart(correlation_figure, use_container_width=True)

with table_tab:
    st.subheader("Building-level records")
    visible_columns = [
        "osm_id",
        "type",
        "designacao_simplificada",
        "area",
        "pop_total",
        "pop_64_mais",
        "numero_servicos_proximos",
        "distancia_media_servicos",
        "cluster_kmeans",
        "latitude",
        "longitude",
    ]
    display_data = filtered[visible_columns].rename(columns=column_label)
    st.dataframe(display_data, use_container_width=True, height=460)

    csv_data = display_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_data,
        file_name="geoinsightlab_filtered_data.csv",
        mime="text/csv",
    )
