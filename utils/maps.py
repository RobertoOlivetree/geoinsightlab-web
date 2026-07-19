from __future__ import annotations

from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st
from shapely import wkt
from shapely.geometry import mapping
from shapely.validation import make_valid

from utils.data_loader import column_label


DEFAULT_POLYGON_COLOUR = [128, 128, 128, 175]


def _rgba_to_css(colour: list[int]) -> str:
    red, green, blue, *alpha = colour
    opacity = (alpha[0] / 255) if alpha else 1.0
    return f"rgba({red}, {green}, {blue}, {opacity:.3f})"


def _continuous_colour(value: float) -> list[int]:
    value = max(0.0, min(1.0, float(value)))
    return [
        int(40 + 205 * value),
        int(165 - 95 * value),
        int(215 - 120 * value),
        190,
    ]


def _valid_geometry(geometry_text: Any):
    if not isinstance(geometry_text, str) or not geometry_text.strip():
        return None

    try:
        geometry = wkt.loads(geometry_text)
    except Exception:
        return None

    if geometry.is_empty:
        return None

    if not geometry.is_valid:
        try:
            geometry = make_valid(geometry)
        except Exception:
            geometry = geometry.buffer(0)

    if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        return None

    return geometry


def _normalise_property_value(value: Any) -> Any:
    """Convert values into JSON-safe Python objects."""

    if isinstance(value, (list, tuple)):
        return [
            item.item() if hasattr(item, "item") else item
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: item.item() if hasattr(item, "item") else item
            for key, item in value.items()
        }

    if hasattr(value, "item"):
        value = value.item()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def _build_geojson(
    dataframe: pd.DataFrame,
    property_columns: tuple[str, ...],
    colour_column: str,
) -> dict:
    """
    Build GeoJSON from the real WKT building footprints.

    This function is intentionally not cached because the dataframe contains
    list-valued colour properties, which Streamlit cannot hash reliably.
    """
    features = []

    selected_columns = list(
        dict.fromkeys(["geometry_wkt", *property_columns, colour_column])
    )
    working = dataframe[selected_columns]

    for row in working.itertuples(index=False, name=None):
        row_data = dict(zip(selected_columns, row))
        geometry = _valid_geometry(row_data.pop("geometry_wkt", None))

        if geometry is None:
            continue

        properties = {
            key: _normalise_property_value(value)
            for key, value in row_data.items()
        }

        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": properties,
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _sample_for_map(
    dataframe: pd.DataFrame,
    maximum_polygons: int,
) -> tuple[pd.DataFrame, bool]:
    if len(dataframe) <= maximum_polygons:
        return dataframe.copy(), False

    return (
        dataframe.sample(maximum_polygons, random_state=42).copy(),
        True,
    )


def _view_state(dataframe: pd.DataFrame) -> pdk.ViewState:
    coordinates = dataframe.dropna(subset=["latitude", "longitude"])

    if coordinates.empty:
        return pdk.ViewState(
            latitude=41.1579,
            longitude=-8.6291,
            zoom=11,
        )

    return pdk.ViewState(
        latitude=float(coordinates["latitude"].mean()),
        longitude=float(coordinates["longitude"].mean()),
        zoom=11.2,
        pitch=0,
    )


def _render_geojson_layer(
    geojson: dict,
    dataframe: pd.DataFrame,
    tooltip: dict,
) -> None:
    if not geojson["features"]:
        st.info(
            "No valid Polygon or MultiPolygon building geometries are available "
            "for the current selection."
        )
        return

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        pickable=True,
        auto_highlight=True,
        filled=True,
        stroked=True,
        get_fill_color="properties._fill_colour",
        get_line_color=[18, 24, 32, 105],
        get_line_width=0.7,
        line_width_min_pixels=0.15,
        line_width_max_pixels=0.8,
        highlight_color=[255, 255, 255, 150],
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=_view_state(dataframe),
        tooltip=tooltip,
        # CARTO dark basemap without labels. This preserves spatial context
        # while reducing visual competition from roads and place names.
        map_style=(
            "https://basemaps.cartocdn.com/gl/"
            "dark-matter-nolabels-gl-style/style.json"
        ),
    )

    st.pydeck_chart(deck, width="stretch")


def render_continuous_polygon_map(
    dataframe: pd.DataFrame,
    value_column: str,
    maximum_polygons: int,
) -> None:
    required = {
        "geometry_wkt",
        "latitude",
        "longitude",
        "osm_id",
        "designacao_simplificada",
        "numero_servicos_proximos",
        "distancia_media_servicos",
        value_column,
    }

    missing = sorted(required.difference(dataframe.columns))
    if missing:
        st.error("Missing map columns: " + ", ".join(missing))
        return

    map_data = dataframe.dropna(
        subset=["geometry_wkt", value_column]
    ).copy()

    if map_data.empty:
        st.info("No observations are available for mapping.")
        return

    map_data, sampled = _sample_for_map(
        map_data,
        maximum_polygons,
    )

    values = pd.to_numeric(
        map_data[value_column],
        errors="coerce",
    )

    valid_values = values.dropna()
    if valid_values.empty:
        st.info(
            "The selected variable does not contain valid numeric values "
            "for mapping."
        )
        return

    minimum = float(valid_values.quantile(0.02))
    maximum = float(valid_values.quantile(0.98))

    if not pd.notna(minimum) or not pd.notna(maximum):
        st.info(
            "The selected variable does not contain sufficient numeric "
            "information for mapping."
        )
        return

    interval = max(maximum - minimum, 1e-9)

    normalised = (
        (values - minimum) / interval
    ).clip(0, 1).fillna(0.5)

    map_data["_fill_colour"] = normalised.apply(
        _continuous_colour
    )

    properties = (
        "osm_id",
        "designacao_simplificada",
        value_column,
        "numero_servicos_proximos",
        "distancia_media_servicos",
        "_fill_colour",
    )

    geojson = _build_geojson(
        map_data,
        properties,
        "_fill_colour",
    )

    value_label = column_label(value_column)

    tooltip = {
        "html": (
            "<b>OSM ID:</b> {osm_id}<br/>"
            "<b>Parish:</b> {designacao_simplificada}<br/>"
            f"<b>{value_label}:</b> {{{value_column}}}<br/>"
            "<b>Nearby services:</b> "
            "{numero_servicos_proximos}<br/>"
            "<b>Mean distance:</b> "
            "{distancia_media_servicos} m"
        )
    }

    _render_geojson_layer(
        geojson,
        map_data,
        tooltip,
    )

    if sampled:
        st.caption(
            f"For performance reasons, the map displays a reproducible sample of "
            f"{len(map_data):,} building footprints. "
            "All statistics and analyses are computed using the complete "
            "filtered dataset."
        )
    else:
        st.caption(
            f"The map displays all {len(map_data):,} building footprints "
            "available for the current selection."
        )

    st.caption(
        f"Colour scale: {minimum:.2f}–{maximum:.2f} "
        "(2nd–98th percentiles)."
    )


def render_categorical_polygon_map(
    dataframe: pd.DataFrame,
    category_column: str,
    colour_map: dict[str, list[int]],
    maximum_polygons: int,
    category_prefix: str = "Cluster",
) -> None:
    required = {
        "geometry_wkt",
        "latitude",
        "longitude",
        "osm_id",
        "designacao_simplificada",
        "numero_servicos_proximos",
        "distancia_media_servicos",
        category_column,
    }

    missing = sorted(required.difference(dataframe.columns))
    if missing:
        st.error("Missing map columns: " + ", ".join(missing))
        return

    map_data = dataframe.dropna(
        subset=["geometry_wkt", category_column]
    ).copy()

    if map_data.empty:
        st.info("No observations are available for mapping.")
        return

    map_data, sampled = _sample_for_map(
        map_data,
        maximum_polygons,
    )

    map_data["_category_label"] = map_data[
        category_column
    ].map(
        lambda value: f"{category_prefix} {int(value)}"
    )

    map_data["_fill_colour"] = map_data[
        "_category_label"
    ].map(
        lambda label: colour_map.get(
            label,
            DEFAULT_POLYGON_COLOUR,
        )
    )

    properties = (
        "osm_id",
        "designacao_simplificada",
        "numero_servicos_proximos",
        "distancia_media_servicos",
        "_category_label",
        "_fill_colour",
    )

    geojson = _build_geojson(
        map_data,
        properties,
        "_fill_colour",
    )

    present_categories = set(
        map_data["_category_label"]
        .dropna()
        .astype(str)
        .unique()
    )

    filtered_colour_map = {
        label: colour
        for label, colour in colour_map.items()
        if label in present_categories
    }

    legend_items = " &nbsp; ".join(
        (
            f'<span style="display:inline-block;'
            f'width:12px;height:12px;'
            f'border-radius:2px;'
            f'background:{_rgba_to_css(colour)};'
            f'margin-right:5px;"></span>{label}'
        )
        for label, colour in filtered_colour_map.items()
    )

    st.markdown(
        legend_items,
        unsafe_allow_html=True,
    )

    tooltip = {
        "html": (
            "<b>OSM ID:</b> {osm_id}<br/>"
            "<b>Parish:</b> {designacao_simplificada}<br/>"
            "<b>Cluster:</b> {_category_label}<br/>"
            "<b>Nearby services:</b> "
            "{numero_servicos_proximos}<br/>"
            "<b>Mean distance:</b> "
            "{distancia_media_servicos} m"
        )
    }

    _render_geojson_layer(
        geojson,
        map_data,
        tooltip,
    )

    if sampled:
        st.caption(
            f"For performance reasons, the map displays a reproducible sample of "
            f"{len(map_data):,} building footprints. "
            "All cluster statistics and analyses are computed using the complete "
            "filtered dataset."
        )
    else:
        st.caption(
            f"The map displays all {len(map_data):,} building footprints "
            "available for the current selection."
        )
