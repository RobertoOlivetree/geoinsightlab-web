import streamlit as st

from utils.data_loader import load_data
from utils.theme import apply_theme


apply_theme()

dataframe = load_data()

st.markdown(
    '''
    <div class="geo-hero">
        <h1>🌍 GEOInsightLab</h1>
        <h3>Data Science for Spatial Literacy</h3>
        <p>
            Plataforma científica interativa associada à investigação
            doutoral sobre Ciência de Dados na Literacia Espacial.
        </p>
    </div>
    ''',
    unsafe_allow_html=True,
)

st.write(
    "Esta primeira versão estável estabelece a base técnica do projeto e "
    "liga a aplicação ao conjunto de dados real do estudo de caso do Porto."
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Edifícios",
    f"{len(dataframe):,}".replace(",", " "),
)
metric_2.metric(
    "Variáveis",
    len(dataframe.columns),
)
metric_3.metric(
    "Freguesias",
    dataframe["designacao_simplificada"].nunique(),
)
metric_4.metric(
    "Unidade espacial",
    "Edifício",
)

st.subheader("Fluxo científico")

st.code(
    "Dados → Exploração → Clustering → IA Explicável → Cidade de 15 Minutos "
    "→ Aprendizagem Supervisionada → Atratividade Urbana "
    "→ Diagnóstico Espacial",
    language=None,
)

st.subheader("Primeiro módulo disponível")

column_1, column_2 = st.columns(2)

with column_1:
    st.markdown(
        '''
        <div class="geo-card">
            <h3>🗺️ Data Exploration</h3>
            <p>
                Exploração estatística e espacial dos 31 873 edifícios,
                com filtros por freguesia, mapas, gráficos e exportação.
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

with column_2:
    st.markdown(
        '''
        <div class="geo-card">
            <h3>🔬 Desenvolvimento faseado</h3>
            <p>
                Os restantes módulos serão integrados e testados
                separadamente para preservar a estabilidade da aplicação.
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Roberto de Oliveira Machado · NOVA FCSH / CICS.NOVA"
)
