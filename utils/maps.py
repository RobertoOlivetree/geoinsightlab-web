import pandas as pd
import pydeck as pdk
import streamlit as st


def render_point_map(
    dataframe: pd.DataFrame,
    value_column: str,
    maximum_points: int,
) -> None:
    map_data = dataframe.dropna(
        subset=["latitude", "longitude", value_column]
    ).copy()

    if map_data.empty:
        st.info("Não existem observações para representar no mapa.")
        return

    if len(map_data) > maximum_points:
        map_data = map_data.sample(
            maximum_points,
            random_state=42,
        )

    values = pd.to_numeric(
        map_data[value_column],
        errors="coerce",
    )

    minimum = float(values.quantile(0.02))
    maximum = float(values.quantile(0.98))
    interval = max(maximum - minimum, 1e-9)

    normalised = (
        (values - minimum) / interval
    ).clip(0, 1).fillna(0.5)

    map_data["_fill_colour"] = normalised.apply(
        lambda value: [
            int(40 + 205 * value),
            int(165 - 95 * value),
            int(215 - 120 * value),
            175,
        ]
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position="[longitude, latitude]",
        get_fill_color="_fill_colour",
        get_radius=24,
        radius_min_pixels=1.5,
        radius_max_pixels=8,
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=float(map_data["latitude"].mean()),
        longitude=float(map_data["longitude"].mean()),
        zoom=11.2,
        pitch=0,
    )

    tooltip = {
        "html": (
            "<b>OSM ID:</b> {osm_id}<br/>"
            "<b>Freguesia:</b> {designacao_simplificada}<br/>"
            f"<b>{value_column}:</b> {{{value_column}}}<br/>"
            "<b>Serviços:</b> {numero_servicos_proximos}<br/>"
            "<b>Distância média:</b> {distancia_media_servicos} m"
        )
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None,
    )

    st.pydeck_chart(
        deck,
        use_container_width=True,
    )
