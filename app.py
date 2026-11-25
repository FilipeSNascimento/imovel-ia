import inspect
import json
import altair as alt
import joblib
import pandas as pd
import streamlit as st
from treinamento import BASE, COLUNAS, treinar

REPO_URL = 'https://github.com/FilipeSNascimento/imovel-ia'
INTEGRANTES = 'Filipe · Letícia · Maurício · Sabrina'

st.set_page_config(page_title='Imóvel IA', page_icon='🏠', layout='centered')


def _largura_total(fn):
    return {'width': 'stretch'} if 'width' in inspect.signature(fn).parameters else {'use_container_width': True}


st.markdown(
    """<style>
.block-container {max-width: 1100px; padding-top: 3rem; padding-bottom: 3rem;}
.hero {
    background: linear-gradient(120deg,#101D32 0%,#142B3C 65%,#123E3C 100%);
    border: 1px solid #284454;
    color: #F1F5F9;
    padding: 36px;
    border-radius: 20px;
    margin-bottom: 24px;
    box-shadow: 0 16px 40px #0003;
}
.hero h1 {color: #F1F5F9; margin: 8px 0; padding: 0; font-size: 48px; letter-spacing: -2px;}
.hero p {color: #B7CAD8; margin: 10px 0 0;}
.hero small {letter-spacing: 2px; color: #5EEAD4; font-weight: 700;}
[data-testid="stMetric"] {background: #111E30; border: 1px solid #26384C; border-radius: 14px; padding: 18px;}
[data-testid="stMetricValue"] {font-size: clamp(19px, 2.5vw, 30px); color: #E7F4F3;}
[data-testid="stForm"] {background: #111E30; border: 1px solid #26384C; border-radius: 18px; padding: 26px;}
[data-testid="stFormSubmitButton"] button {
    background: #5EEAD4; color: #09231F; border: 0; font-weight: 700; min-height: 48px; border-radius: 10px;
}
[data-testid="stFormSubmitButton"] button:hover {background: #99F6E4; color: #09231F;}
[data-testid="stExpander"] {border: 1px solid #26384C; border-radius: 14px; background: #111E30;}
.footer-box {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 8px; color: #94A3B8; font-size: 0.875rem; padding-top: 4px;
}
.footer-box a {color: #5EEAD4; text-decoration: none; font-weight: 600;}
.footer-box a:hover {text-decoration: underline;}
</style>
<div class="hero">
    <small>INTELIGÊNCIA IMOBILIÁRIA / MACHINE LEARNING</small>
    <h1>Imóvel IA</h1>
    <p>Descubra o valor estimado de um imóvel e compare com o preço anunciado.</p>
    <p>Ames, Iowa · Estados Unidos · Dados históricos de 2006–2010</p>
</div>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def carregar():
    modelo_path = BASE / 'modelo.joblib'
    metricas_path = BASE / 'metricas.json'
    if modelo_path.exists() and metricas_path.exists():
        try:
            modelo_salvo = joblib.load(modelo_path)
            metricas_salvas = json.loads(metricas_path.read_text(encoding='utf-8'))
            amostra = pd.DataFrame([[1500.0, 3, 2, 2000, 7, 2]], columns=COLUNAS)
            modelo_salvo.predict(amostra)
            return modelo_salvo, metricas_salvas
        except Exception:
            pass
    metricas_geradas = treinar()
    return joblib.load(modelo_path), metricas_geradas


try:
    with st.spinner('Preparando modelo de Machine Learning...'):
        modelo, metricas = carregar()
except Exception as erro:
    st.error(
        f'Não foi possível preparar o modelo: {erro}. '
        'Verifique a conexão ou coloque o CSV AmesHousing como dataset.csv na pasta do projeto.'
    )
    st.stop()

col_a, col_b, col_c = st.columns(3)
col_a.metric('Imóveis na base', f'{metricas["registros"]:,}'.replace(',', '.'))
col_b.metric('Modelo selecionado', metricas['modelo'])
col_c.metric('R² no teste', f'{metricas["r2"]:.3f}')

st.caption('01 / AVALIAÇÃO  ·  Preencha as características para consultar o modelo treinado.')

with st.form('imovel'):
    st.subheader('Características do imóvel')
    a, b = st.columns(2)
    area = a.number_input('Área habitável (m²)', 31.0, 524.0, 150.0)
    quartos = b.number_input('Quartos', 0, 8, 3)
    banheiros = a.number_input('Banheiros', 0, 4, 2)
    ano = b.number_input('Ano de construção', 1872, 2010, 2000)
    qualidade = a.slider('Qualidade do imóvel (1 a 10)', 1, 10, 7)
    garagem = b.number_input('Vagas na garagem', 0, 5, 2)
    anuncio = st.number_input('Preço anunciado (US$)', min_value=1.0, value=250000.0, step=1000.0)
    calcular = st.form_submit_button('Estimar preço', **_largura_total(st.form_submit_button))

if calcular:
    entrada = pd.DataFrame([[area / 0.09290304, quartos, banheiros, ano, qualidade, garagem]], columns=COLUNAS)
    estimativa = float(modelo.predict(entrada)[0])
    if estimativa <= 0:
        st.warning('Estimativa fora da faixa válida. Altere as características.')
    else:
        diferenca = (anuncio - estimativa) / estimativa * 100
        st.subheader('02 / Resultado da avaliação')
        r1, r2, r3 = st.columns(3)
        r1.metric('Estimativa da IA', f'US$ {estimativa:,.0f}')
        r2.metric('Preço anunciado', f'US$ {anuncio:,.0f}', f'{diferenca:+.1f}% vs. estimativa', delta_color='off')
        r3.metric('Diferença em dólares', f'US$ {anuncio - estimativa:,.0f}')

        comparacao = pd.DataFrame({'Valor': [anuncio, estimativa], 'Tipo': ['Anunciado', 'Estimado pela IA']})
        grafico = (
            alt.Chart(comparacao)
            .mark_bar(cornerRadiusEnd=7, size=48)
            .encode(
                x=alt.X('Valor:Q', title='Preço (US$)', axis=alt.Axis(format=',.0f')),
                y=alt.Y('Tipo:N', title=None, sort=['Anunciado', 'Estimado pela IA']),
                color=alt.Color(
                    'Tipo:N',
                    scale=alt.Scale(domain=['Anunciado', 'Estimado pela IA'], range=['#8299B8', '#5EEAD4']),
                    legend=None,
                ),
                tooltip=['Tipo', alt.Tooltip('Valor:Q', format=',.2f')],
            )
            .properties(height=170)
        )
        st.altair_chart(grafico, **_largura_total(st.altair_chart))

        if diferenca > 0:
            st.warning(f'O preço anunciado está {diferenca:.1f}% acima da estimativa da IA.')
        elif diferenca < 0:
            st.success(f'O preço anunciado está {abs(diferenca):.1f}% abaixo da estimativa da IA.')
        else:
            st.info('O preço anunciado coincide com a estimativa.')

with st.expander('Sobre o modelo e resultados'):
    st.markdown('**Como a IA foi avaliada**')
    m1, m2, m3 = st.columns(3)
    m1.metric('MAE', f'US$ {metricas["mae"]:,.0f}')
    m2.metric('RMSE', f'US$ {metricas["rmse"]:,.0f}')
    m3.metric('R²', f'{metricas["r2"]:.3f}')
    st.caption('MAE: erro absoluto médio. RMSE: penaliza erros maiores. R²: proporção da variabilidade explicada pelo modelo no teste.')

    tabela = pd.DataFrame(list(metricas['mae_validacao'].items()), columns=['Algoritmo', 'MAE'])
    st.markdown('**Comparação na validação — menor erro é melhor**')
    grafico_val = (
        alt.Chart(tabela)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X('MAE:Q', title='Erro médio (US$)'),
            y=alt.Y('Algoritmo:N', title=None),
            color=alt.Color('Algoritmo:N', scale=alt.Scale(range=['#5EEAD4', '#8299B8']), legend=None),
            tooltip=['Algoritmo', alt.Tooltip('MAE:Q', format=',.2f')],
        )
        .properties(height=110)
    )
    st.altair_chart(grafico_val, **_largura_total(st.altair_chart))
    st.dataframe(tabela.style.format({'MAE': 'US$ {:,.2f}'}), hide_index=True, **_largura_total(st.dataframe))
    st.write(
        f'O modelo {metricas["modelo"]} foi escolhido por apresentar o menor MAE na validação. '
        'O desempenho final foi medido em dados de teste separados.'
    )
    st.markdown('**Etapas:** tratamento de valores ausentes → treinamento → validação → teste → previsão.')

st.caption('Os campos de área, quartos e banheiros consideram apenas os pavimentos acima do nível do terreno; banheiros são completos, conforme a base original.')
st.divider()
st.caption('Projeto acadêmico. Valores históricos de 2006–2010 em dólares; não representam o mercado atual nem uma avaliação profissional.')
st.markdown(
    f"""<div class="footer-box">
    <span><strong>Integrantes:</strong> {INTEGRANTES}</span>
    <a href="{REPO_URL}" target="_blank">Repositório no GitHub ↗</a>
</div>""",
    unsafe_allow_html=True,
)
