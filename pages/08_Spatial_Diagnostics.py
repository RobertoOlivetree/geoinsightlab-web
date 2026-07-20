"""GEOInsightLab — Spatial Diagnostics Streamlit page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_data
from utils.theme import apply_theme, page_header, scientific_note

try:
    from utils.maps import render_categorical_polygon_map
except ImportError:
    render_categorical_polygon_map = None


apply_theme()

page_header(
    "Spatial Diagnostics",
    "Explore the spatial autocorrelation of urban attractiveness and identify "
    "statistically significant spatial clusters across Porto.",
)

scientific_note(
    "This module evaluates whether urban-attractiveness patterns are spatially "
    "clustered using global and local indicators of spatial autocorrelation. "
    "It complements the Urban Attractiveness and SHAP Analysis modules."
)


THESIS_SENSITIVITY = pd.DataFrame(
    {
        "Spatial unit": [
            "Buildings",
            "Buildings",
            "Buildings",
            "BGRI",
            "100 m grid",
            "250 m grid",
            "500 m grid",
        ],
        "Neighbourhood structure": [
            "KNN (8)",
            "KNN (12)",
            "KNN (16)",
            "Contiguity",
            "Contiguity",
            "Contiguity",
            "Contiguity",
        ],
        "Moran's I": [
            0.984,
            0.981,
            0.978,
            0.927,
            0.945,
            0.868,
            0.728,
        ],
        "p-value": [
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
        ],
    }
)

LISA_ALIASES = [
    "lisa_cluster",
    "LISA",
    "lisa",
    "cluster_lisa",
    "moran_local_cluster",
]

SIGNIFICANCE_ALIASES = [
    "lisa_significant",
    "lisa_significance",
    "moran_local_significant",
    "significant",
]

LISA_LABELS = {
    0: "Not significant",
    1: "High–High",
    2: "Low–High",
    3: "Low–Low",
    4: "High–Low",
    "0": "Not significant",
    "1": "High–High",
    "2": "Low–High",
    "3": "Low–Low",
    "4": "High–Low",
    "HH": "High–High",
    "HL": "High–Low",
    "LH": "Low–High",
    "LL": "Low–Low",
    "High-High": "High–High",
    "High-Low": "High–Low",
    "Low-High": "Low–High",
    "Low-Low": "Low–Low",
    "Not Significant": "Not significant",
}


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    return next(
        (
            field
            for field in candidates
            if field in dataframe.columns
        ),
        None,
    )


def prepare_lisa_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    lisa_field = first_existing_column(
        dataframe,
        LISA_ALIASES,
    )

    if lisa_field is None:
        return dataframe.copy(), None

    result = dataframe.copy()

    result["lisa_class"] = (
        result[lisa_field]
        .map(LISA_LABELS)
        .fillna(
            result[lisa_field].astype(str)
        )
    )

    significance_field = first_existing_column(
        result,
        SIGNIFICANCE_ALIASES,
    )

    if significance_field is not None:
        significant = result[
            significance_field
        ].astype(bool)

        result.loc[
            ~significant,
            "lisa_class",
        ] = "Not significant"

    return result, "lisa_class"


try:
    data = load_data()
except Exception as error:
    st.error(
        f"The research dataset could not be loaded: {error}"
    )
    st.stop()


data, lisa_column = prepare_lisa_data(
    data
)


with st.sidebar:
    st.subheader(
        "Spatial diagnostics controls"
    )

    selected_unit = st.selectbox(
        "Spatial unit",
        [
            "Buildings",
            "BGRI",
            "100 m grid",
            "250 m grid",
            "500 m grid",
        ],
    )

    available_structures = (
        THESIS_SENSITIVITY.loc[
            THESIS_SENSITIVITY[
                "Spatial unit"
            ].eq(selected_unit),
            "Neighbourhood structure",
        ]
        .drop_duplicates()
        .tolist()
    )

    selected_structure = st.selectbox(
        "Neighbourhood structure",
        available_structures,
    )


selected_result = THESIS_SENSITIVITY[
    THESIS_SENSITIVITY[
        "Spatial unit"
    ].eq(selected_unit)
    & THESIS_SENSITIVITY[
        "Neighbourhood structure"
    ].eq(selected_structure)
].iloc[0]


metric_columns = st.columns(4)

metric_columns[0].metric(
    "Spatial unit",
    selected_unit,
)

metric_columns[1].metric(
    "Neighbourhood",
    selected_structure,
)

metric_columns[2].metric(
    "Global Moran's I",
    f"{selected_result['Moran\'s I']:.3f}",
)

metric_columns[3].metric(
    "Permutation p-value",
    f"{selected_result['p-value']:.3f}",
)


(
    global_tab,
    lisa_tab,
    hotspots_tab,
    sensitivity_tab,
    residuals_tab,
    methodology_tab,
) = st.tabs(
    [
        "Global Moran's I",
        "LISA clusters",
        "Hotspots and coldspots",
        "Sensitivity analysis",
        "Residual diagnostics",
        "Methodology",
    ]
)


with global_tab:
    st.subheader(
        "Global Spatial Autocorrelation"
    )

    global_chart_data = (
        THESIS_SENSITIVITY[
            THESIS_SENSITIVITY[
                "Spatial unit"
            ].eq(selected_unit)
        ]
        .sort_values(
            "Moran's I",
            ascending=True,
        )
    )

    chart = px.bar(
        global_chart_data,
        x="Moran's I",
        y="Neighbourhood structure",
        orientation="h",
        title=(
            f"Global Moran's I — {selected_unit}"
        ),
        text="Moran's I",
    )

    chart.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    chart.update_layout(
        xaxis_title="Moran's I",
        yaxis_title="",
    )

    st.plotly_chart(
        chart,
        width="stretch",
    )

    st.info(
        "The positive and high Moran's I values indicate strong spatial "
        "clustering: buildings or spatial units with similar attractiveness "
        "values tend to occur near one another."
    )


with lisa_tab:
    st.subheader(
        "Local Moran's I Cluster Map"
    )

    if (
        lisa_column is not None
        and render_categorical_polygon_map
        is not None
    ):
        render_categorical_polygon_map(
            dataframe=data,
            category_column=lisa_column,
        )

        lisa_counts = (
            data[lisa_column]
            .value_counts(
                dropna=False
            )
            .rename_axis(
                "LISA class"
            )
            .reset_index(
                name="Buildings"
            )
        )

        st.dataframe(
            lisa_counts,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "The current dataset does not contain validated Local Moran's I "
            "classes. Add a field such as `lisa_cluster` to activate the "
            "interactive LISA map."
        )

        st.markdown(
            """
The map will distinguish:

- **High–High** — high values surrounded by high values;
- **Low–Low** — low values surrounded by low values;
- **High–Low** — high-value spatial outliers;
- **Low–High** — low-value spatial outliers;
- **Not significant** — no statistically significant local association.
            """
        )


with hotspots_tab:
    st.subheader(
        "Hotspots and Coldspots"
    )

    if lisa_column is not None:
        class_counts = (
            data[lisa_column]
            .value_counts()
        )

        hotspots = int(
            class_counts.get(
                "High–High",
                0,
            )
        )
        coldspots = int(
            class_counts.get(
                "Low–Low",
                0,
            )
        )
        spatial_outliers = int(
            class_counts.get(
                "High–Low",
                0,
            )
            + class_counts.get(
                "Low–High",
                0,
            )
        )

        hotspot_columns = st.columns(3)

        hotspot_columns[0].metric(
            "High–High hotspots",
            f"{hotspots:,}",
        )
        hotspot_columns[1].metric(
            "Low–Low coldspots",
            f"{coldspots:,}",
        )
        hotspot_columns[2].metric(
            "Spatial outliers",
            f"{spatial_outliers:,}",
        )

        hotspot_table = pd.DataFrame(
            {
                "LISA class": [
                    "High–High",
                    "Low–Low",
                    "High–Low",
                    "Low–High",
                    "Not significant",
                ],
                "Buildings": [
                    int(
                        class_counts.get(
                            category,
                            0,
                        )
                    )
                    for category in [
                        "High–High",
                        "Low–Low",
                        "High–Low",
                        "Low–High",
                        "Not significant",
                    ]
                ],
            }
        )

        hotspot_chart = px.bar(
            hotspot_table,
            x="LISA class",
            y="Buildings",
            title=(
                "Distribution of Local Moran's I classes"
            ),
        )

        st.plotly_chart(
            hotspot_chart,
            width="stretch",
        )
    else:
        st.info(
            "Hotspot and coldspot counts will be calculated after validated "
            "LISA classes are added to the research dataset."
        )


with sensitivity_tab:
    st.subheader(
        "Sensitivity Across Spatial Configurations"
    )

    sensitivity_chart = px.line(
        THESIS_SENSITIVITY,
        x="Spatial unit",
        y="Moran's I",
        color="Neighbourhood structure",
        markers=True,
        title=(
            "Sensitivity of Global Moran's I"
        ),
    )

    sensitivity_chart.update_layout(
        xaxis_title="Spatial unit",
        yaxis_title="Moran's I",
        legend_title=(
            "Neighbourhood structure"
        ),
    )

    st.plotly_chart(
        sensitivity_chart,
        width="stretch",
    )

    display_sensitivity = (
        THESIS_SENSITIVITY.copy()
    )

    display_sensitivity[
        "Moran's I"
    ] = display_sensitivity[
        "Moran's I"
    ].round(3)

    display_sensitivity[
        "p-value"
    ] = display_sensitivity[
        "p-value"
    ].map(
        lambda value: (
            "< 0.001"
            if value <= 0.001
            else f"{value:.3f}"
        )
    )

    st.dataframe(
        display_sensitivity,
        width="stretch",
        hide_index=True,
    )

    st.info(
        "The decrease in Moran's I as the aggregation unit becomes larger "
        "illustrates the sensitivity of spatial statistics to scale and "
        "aggregation. Nevertheless, positive clustering remains consistent "
        "across every tested configuration."
    )


with residuals_tab:
    st.subheader(
        "Residual Spatial Diagnostics"
    )

    st.info(
        "This section is reserved for the Global Moran's I and Local Moran's I "
        "analysis of the standardised XGBoost residuals. Keeping the spatial "
        "residual analysis here avoids duplication with the SHAP Analysis page."
    )

    st.markdown(
        """
The residual workflow should contain:

1. standardised model residuals;
2. Global Moran's I with permutation inference;
3. Local Moran's I classification;
4. a residual LISA cluster map;
5. interpretation of any remaining unexplained spatial structure.
        """
    )


with methodology_tab:
    st.subheader(
        "Analytical Workflow"
    )

    st.markdown(
        """
The module follows the spatial-statistical workflow adopted in the doctoral
research:

1. **Global Moran's I** evaluates whether the complete spatial pattern is
   clustered, dispersed or random.
2. **Local Moran's I** identifies statistically significant local clusters and
   spatial outliers.
3. **High–High** and **Low–Low** observations represent local concentrations of
   similar values.
4. **High–Low** and **Low–High** observations represent local spatial outliers.
5. Statistical significance is assessed through permutation inference.
6. Sensitivity is evaluated across alternative neighbourhood structures and
   spatial aggregation units.
7. The selected building-level specification uses a connected K-nearest
   neighbours structure, while the aggregation analysis evaluates BGRI and
   regular grids of 100, 250 and 500 metres.

### Interpretation limits

Spatial autocorrelation identifies geographical association, not causality.
Results depend on the selected spatial unit, neighbourhood definition,
standardisation procedure and significance threshold.
        """
    )

    st.download_button(
        "Download sensitivity results",
        data=THESIS_SENSITIVITY.to_csv(
            index=False
        ).encode(
            "utf-8"
        ),
        file_name=(
            "geoinsightlab_spatial_diagnostics.csv"
        ),
        mime="text/csv",
    )
