"""GEOInsightLab — 15-Minute City Streamlit page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import column_label, load_data
from utils.maps import render_categorical_polygon_map
from utils.theme import apply_theme, page_header, scientific_note


apply_theme()
page_header(
    "15-Minute City",
    "Assess which buildings have pedestrian access to essential urban services within the 1.5 km network threshold used in the thesis.",
)
scientific_note(
    "This module uses the service-access indicators already stored in the research dataset. "
    "A building is considered covered when at least one establishment of the selected service category is "
    "available within the 1.5 km pedestrian-network threshold."
)

SERVICE_COLUMNS = {
    "Health centres": "Centro Saude",
    "Pharmacies": "Farmacias",
    "Hospitals": "Hospitais",
    "Supermarkets": "Supermercados",
    "Banks": "Bancos",
    "Post offices": "CTT",
    "Parks and gardens": "Parques e jardins",
}

ACCESS_COLOURS = {
    "No access": [214, 39, 40, 235],
    "Access": [46, 139, 87, 235],
}

ACCESS_LABELS = {
    0: "No access",
    1: "Access",
}

DIVERSITY_COLOURS = {
    "Other buildings": [127, 127, 127, 210],
    "Priority": [214, 39, 40, 235],
}

PRIORITY_LABELS = {
    0: "Other buildings",
    1: "Priority",
}


def numeric_series(dataframe: pd.DataFrame, field: str) -> pd.Series:
    return pd.to_numeric(dataframe[field], errors="coerce").fillna(0)


def coverage_for_services(
    dataframe: pd.DataFrame,
    service_fields: list[str],
) -> pd.Series:
    values = pd.concat(
        [numeric_series(dataframe, field) > 0 for field in service_fields],
        axis=1,
    )
    return values.all(axis=1)


def render_access_map(
    dataframe: pd.DataFrame,
    value_column: str,
    maximum_polygons: int,
    colours: dict[str, list[int]],
    labels: dict[int, str],
    category_title: str,
) -> None:
    render_categorical_polygon_map(
        dataframe=dataframe,
        category_column=value_column,
        colour_map=colours,
        maximum_polygons=maximum_polygons,
        category_prefix=category_title,
        label_map=labels,
    )


def weighted_population_without_access(
    dataframe: pd.DataFrame,
    access: pd.Series,
) -> float:
    population = numeric_series(dataframe, "pop_64_mais")
    return float(population.loc[~access].sum())


@st.cache_data(show_spinner=False)
def service_gap_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    total_buildings = len(dataframe)
    total_population = numeric_series(dataframe, "pop_64_mais").sum()

    rows = []
    for label, field in SERVICE_COLUMNS.items():
        access = numeric_series(dataframe, field) > 0
        buildings_without = int((~access).sum())
        population_without = weighted_population_without_access(dataframe, access)

        rows.append(
            {
                "Service": label,
                "Population aged 65+ without access": population_without,
                "Population without access (%)": (
                    100 * population_without / total_population
                    if total_population > 0
                    else np.nan
                ),
                "Buildings without access": buildings_without,
                "Buildings without access (%)": (
                    100 * buildings_without / total_buildings
                    if total_buildings > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "Buildings without access (%)",
        ascending=False,
    )


@st.cache_data(show_spinner=False)
def parish_gap_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    records = []

    for parish, group in dataframe.groupby("designacao_simplificada"):
        for label, field in SERVICE_COLUMNS.items():
            access = numeric_series(group, field) > 0
            records.append(
                {
                    "Parish": parish,
                    "Service": label,
                    "Buildings without access (%)": 100 * (~access).mean(),
                    "Population aged 65+ without access": weighted_population_without_access(
                        group,
                        access,
                    ),
                }
            )

    return pd.DataFrame(records)


try:
    data = load_data()
except Exception as error:
    st.error(f"The research dataset could not be loaded: {error}")
    st.stop()

required = {
    "osm_id",
    "designacao_simplificada",
    "pop_64_mais",
    "numero_servicos_proximos",
    "distancia_media_servicos",
    *SERVICE_COLUMNS.values(),
}

missing = sorted(required.difference(data.columns))
if missing:
    st.error(
        "The dataset is missing fields required by the 15-minute-city module: "
        + ", ".join(missing)
    )
    st.stop()

with st.sidebar:
    st.subheader("15-Minute City controls")

    parishes = sorted(
        data["designacao_simplificada"].dropna().astype(str).unique().tolist()
    )
    selected_parishes = st.multiselect(
        "Parishes",
        options=parishes,
        default=parishes,
    )

    selected_service_label = st.selectbox(
        "Service",
        options=list(SERVICE_COLUMNS),
        index=0,
        help="Select one of the seven service categories stored in the research dataset.",
    )

    maximum_polygons = st.slider(
        "Maximum building polygons on the map",
        min_value=1_000,
        max_value=15_000,
        value=10_000,
        step=1_000,
    )

filtered = data[
    data["designacao_simplificada"].astype(str).isin(selected_parishes)
].copy()

if filtered.empty:
    st.warning("Select at least one parish.")
    st.stop()

scenario_label = str(selected_service_label)
selected_service_field = SERVICE_COLUMNS[scenario_label]
covered = numeric_series(filtered, selected_service_field) > 0

filtered["access_status"] = covered.astype(int)

total_buildings = len(filtered)
covered_buildings = int(covered.sum())
coverage_share = 100 * covered.mean()

population_65 = numeric_series(filtered, "pop_64_mais")
covered_population = float(population_65.loc[covered].sum())
total_population = float(population_65.sum())

population_coverage_share = (
    100 * covered_population / total_population
    if total_population > 0
    else np.nan
)

metrics = st.columns(5)
metrics[0].metric("Buildings", f"{total_buildings:,}")
metrics[1].metric("Buildings with access", f"{covered_buildings:,}")
metrics[2].metric("Building coverage", f"{coverage_share:.1f}%")
metrics[3].metric(
    "Population aged 65+ with access",
    f"{covered_population:,.0f}",
)
metrics[4].metric(
    "Population aged 65+ coverage",
    f"{population_coverage_share:.1f}%",
)

map_tab, service_tab, parish_tab, priority_tab, methodology_tab = st.tabs(
    [
        "Coverage map",
        "Service gaps",
        "Parish comparison",
        "Priority buildings",
        "Methodology",
    ]
)

with map_tab:
    st.subheader(f"Pedestrian coverage — {scenario_label}")
    st.caption(
        "Access indicates that at least one establishment of the selected service is "
        "available within the 1.5 km pedestrian-network threshold. No access identifies "
        "a coverage gap. Real Polygon or MultiPolygon building footprints are used."
    )

    render_access_map(
        filtered,
        "access_status",
        maximum_polygons,
        ACCESS_COLOURS,
        ACCESS_LABELS,
        "Access",
    )

    distribution = pd.DataFrame(
        {
            "Status": ["With access", "Without access"],
            "Buildings": [
                covered_buildings,
                total_buildings - covered_buildings,
            ],
        }
    )
    distribution["Share (%)"] = (
        100 * distribution["Buildings"] / distribution["Buildings"].sum()
    )

    figure = px.bar(
        distribution,
        x="Status",
        y="Buildings",
        text="Share (%)",
        title="Building coverage",
    )
    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    figure.update_layout(
        xaxis_title="",
        yaxis_title="Number of buildings",
    )
    st.plotly_chart(figure, width="stretch")

with service_tab:
    st.subheader("Coverage gaps by service")

    gaps = service_gap_table(filtered)
    display_gaps = gaps.copy()

    for field in [
        "Population aged 65+ without access",
        "Buildings without access",
    ]:
        display_gaps[field] = display_gaps[field].round(0).astype(int)

    for field in [
        "Population without access (%)",
        "Buildings without access (%)",
    ]:
        display_gaps[field] = display_gaps[field].round(2)

    st.dataframe(
        display_gaps,
        width="stretch",
        hide_index=True,
    )

    gap_chart = px.bar(
        gaps.sort_values("Buildings without access (%)", ascending=True),
        x="Buildings without access (%)",
        y="Service",
        orientation="h",
        title="Buildings without pedestrian access by service",
    )
    gap_chart.update_layout(xaxis_ticksuffix="%")
    st.plotly_chart(gap_chart, width="stretch")

    population_chart = px.bar(
        gaps.sort_values("Population without access (%)", ascending=True),
        x="Population without access (%)",
        y="Service",
        orientation="h",
        title="Population aged 65+ without pedestrian access by service",
    )
    population_chart.update_layout(xaxis_ticksuffix="%")
    st.plotly_chart(population_chart, width="stretch")

with parish_tab:
    st.subheader("Accessibility gaps by parish")

    parish_gaps = parish_gap_table(filtered)
    heatmap_data = parish_gaps.pivot(
        index="Parish",
        columns="Service",
        values="Buildings without access (%)",
    )

    heatmap = px.imshow(
        heatmap_data,
        text_auto=".1f",
        aspect="auto",
        title="Share of buildings without access (%)",
    )
    heatmap.update_layout(
        xaxis_title="Service",
        yaxis_title="Parish",
        coloraxis_colorbar_title="Without access (%)",
    )
    st.plotly_chart(heatmap, width="stretch")

    selected_parish_service = st.selectbox(
        "Service shown in the parish ranking",
        list(SERVICE_COLUMNS),
        index=0,
    )

    parish_ranking = parish_gaps[
        parish_gaps["Service"] == selected_parish_service
    ].sort_values(
        "Buildings without access (%)",
        ascending=True,
    )

    ranking_chart = px.bar(
        parish_ranking,
        x="Buildings without access (%)",
        y="Parish",
        orientation="h",
        title=f"Coverage gaps for {selected_parish_service}",
    )
    ranking_chart.update_layout(xaxis_ticksuffix="%")
    st.plotly_chart(ranking_chart, width="stretch")

with priority_tab:
    st.subheader("Low diversity and high mean distance")

    if "numero_categorias_proximas" in filtered.columns:
        category_count = numeric_series(
            filtered,
            "numero_categorias_proximas",
        )
    else:
        category_count = pd.concat(
            [
                numeric_series(filtered, field) > 0
                for field in SERVICE_COLUMNS.values()
            ],
            axis=1,
        ).sum(axis=1)

    mean_distance = numeric_series(
        filtered,
        "distancia_media_servicos",
    )
    distance_threshold = float(mean_distance.quantile(0.75))

    priority = (
        (category_count < 5)
        & (mean_distance > distance_threshold)
    )

    filtered["priority_status"] = priority.astype(int)

    priority_buildings = int(priority.sum())
    priority_population = float(
        population_65.loc[priority].sum()
    )

    priority_metrics = st.columns(4)
    priority_metrics[0].metric(
        "Priority buildings",
        f"{priority_buildings:,}",
    )
    priority_metrics[1].metric(
        "Share of buildings",
        f"{100 * priority.mean():.1f}%",
    )
    priority_metrics[2].metric(
        "Population aged 65+",
        f"{priority_population:,.0f}",
    )
    priority_metrics[3].metric(
        "Distance threshold",
        f"{distance_threshold:,.1f} m",
    )

    st.caption(
        "Priority 1 identifies buildings with fewer than five service categories and "
        "a mean service distance above the 75th percentile. Priority 0 identifies the "
        "remaining buildings."
    )

    render_access_map(
        filtered,
        "priority_status",
        maximum_polygons,
        DIVERSITY_COLOURS,
        PRIORITY_LABELS,
        "Priority status",
    )

    priority_by_parish = (
        filtered.assign(priority=priority)
        .groupby("designacao_simplificada")
        .agg(
            Buildings=("osm_id", "count"),
            Priority_buildings=("priority", "sum"),
            Population_65_plus=("pop_64_mais", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "designacao_simplificada": "Parish",
            }
        )
    )

    priority_by_parish["Priority buildings (%)"] = (
        100
        * priority_by_parish["Priority_buildings"]
        / priority_by_parish["Buildings"]
    )

    priority_chart = px.bar(
        priority_by_parish.sort_values(
            "Priority buildings (%)",
            ascending=True,
        ),
        x="Priority buildings (%)",
        y="Parish",
        orientation="h",
        title="Priority buildings by parish",
    )
    priority_chart.update_layout(xaxis_ticksuffix="%")
    st.plotly_chart(priority_chart, width="stretch")

with methodology_tab:
    st.subheader("Analytical scope")
    st.markdown(
        """
### Coverage criterion

The thesis operationalises the 15-minute-city concept through a **1.5 km pedestrian-network threshold**. A building is covered for a service when at least one establishment of that category is available within that network distance.

### Service selection

The module analyses one service category at a time, using the seven indicators stored in the research dataset:

- health centres;
- pharmacies;
- hospitals;
- supermarkets;
- banks;
- post offices;
- parks and gardens.

Hospitals are included in the same way as the remaining service categories.

### Priority buildings

The priority criterion reproduces the thesis definition: fewer than five available service categories and a mean distance to services above the 75th percentile.

### Interpretation limits

The module visualises precomputed service counts associated with the 1.5 km pedestrian-network analysis. It does not recalculate the pedestrian graph in each Streamlit session and does not represent a straight-line buffer.
        """
    )

    st.subheader("Service indicators")
    st.dataframe(
        pd.DataFrame(
            {
                "Service": list(SERVICE_COLUMNS),
                "Dataset field": list(SERVICE_COLUMNS.values()),
                "Display label": [
                    column_label(field)
                    for field in SERVICE_COLUMNS.values()
                ],
            }
        ),
        width="stretch",
        hide_index=True,
    )

    download_fields = [
        field
        for field in [
            "osm_id",
            "designacao_simplificada",
            "pop_64_mais",
            "numero_servicos_proximos",
            "distancia_media_servicos",
            *SERVICE_COLUMNS.values(),
            "access_status",
            "priority_status",
            "latitude",
            "longitude",
        ]
        if field in filtered.columns
    ]

    st.download_button(
        "Download 15-minute-city results",
        data=filtered[download_fields].to_csv(index=False).encode("utf-8"),
        file_name="geoinsightlab_15_minute_city.csv",
        mime="text/csv",
    )
