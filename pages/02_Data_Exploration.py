import plotly.express as px
import streamlit as st

from utils.data_loader import (
    filter_by_parish,
    load_data,
    numeric_columns,
)
from utils.maps import render_point_map
from utils.theme import (
    apply_theme,
    page_header,
    scientific_note,
)


apply_theme()

page_header(
    "Data Exploration",
    "Exploração estatística e espacial do conjunto de dados real do Porto.",
)

dataframe = load_data()

parish_options = sorted(
    dataframe["designacao_simplificada"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

number_columns = numeric_columns(dataframe)

preferred_variable = "numero_servicos_proximos"
default_variable_index = (
    number_columns.index(preferred_variable)
    if preferred_variable in number_columns
    else 0
)

with st.sidebar:
    st.header("Filtros")

    selected_parishes = st.multiselect(
        "Freguesias",
        options=parish_options,
        default=parish_options,
    )

    selected_variable = st.selectbox(
        "Variável",
        options=number_columns,
        index=default_variable_index,
    )

    maximum_points = st.slider(
        "Número máximo de pontos no mapa",
        min_value=2_000,
        max_value=20_000,
        value=10_000,
        step=1_000,
    )

filtered = filter_by_parish(
    dataframe,
    selected_parishes,
)

scientific_note(
    "O conjunto de dados contém 31 873 registos de edifícios e 27 "
    "variáveis, incluindo atributos demográficos, indicadores de acesso "
    "a serviços, resultados de clustering e coordenadas geográficas."
)

if filtered.empty:
    st.warning(
        "Seleciona pelo menos uma freguesia para visualizar os resultados."
    )
    st.stop()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Edifícios",
    f"{len(filtered):,}".replace(",", " "),
)
metric_2.metric(
    "Freguesias",
    filtered["designacao_simplificada"].nunique(),
)
metric_3.metric(
    "Média de serviços",
    f"{filtered['numero_servicos_proximos'].mean():.2f}",
)
metric_4.metric(
    "Distância média",
    f"{filtered['distancia_media_servicos'].mean():.1f} m",
)

st.subheader("Distribuição espacial")

render_point_map(
    filtered,
    value_column=selected_variable,
    maximum_points=maximum_points,
)

chart_column_1, chart_column_2 = st.columns(2)

with chart_column_1:
    distribution_figure = px.histogram(
        filtered,
        x=selected_variable,
        nbins=40,
        marginal="box",
        title=f"Distribuição de {selected_variable}",
    )

    st.plotly_chart(
        distribution_figure,
        use_container_width=True,
    )

with chart_column_2:
    parish_summary = (
        filtered.groupby(
            "designacao_simplificada",
            as_index=False,
        )[selected_variable]
        .mean()
        .sort_values(
            selected_variable,
            ascending=True,
        )
    )

    parish_figure = px.bar(
        parish_summary,
        x=selected_variable,
        y="designacao_simplificada",
        orientation="h",
        title=f"Média de {selected_variable} por freguesia",
    )

    st.plotly_chart(
        parish_figure,
        use_container_width=True,
    )

st.subheader("Estatística descritiva")

descriptive_statistics = (
    filtered[number_columns]
    .describe()
    .transpose()
    .round(3)
)

st.dataframe(
    descriptive_statistics,
    use_container_width=True,
)

st.subheader("Dados")

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

st.dataframe(
    filtered[visible_columns],
    use_container_width=True,
    height=430,
)

csv_data = filtered.to_csv(
    index=False,
).encode("utf-8")

st.download_button(
    label="Descarregar dados filtrados em CSV",
    data=csv_data,
    file_name="geoinsightlab_dados_filtrados.csv",
    mime="text/csv",
)
