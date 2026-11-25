# Imóvel IA

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

Projeto final da disciplina de **Inteligência Artificial**. Aplicação web para estimar o preço histórico de venda de imóveis residenciais por meio de aprendizado supervisionado e comparar a previsão da IA com o valor anunciado.

## Integrantes

- **Filipe**
- **Letícia**
- **Maurício**
- **Sabrina**

---

## Como executar no Windows

1. Certifique-se de ter o **Python 3.10+** instalado (marque **Add python.exe to PATH** durante a instalação no Windows).
2. Baixe ou extraia a pasta `imovel-ia`.
3. Dê dois cliques em **`iniciar.bat`** (ou execute os comandos abaixo no terminal dentro da pasta `imovel-ia`):

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. Caso o navegador não abra automaticamente, acesse **[http://localhost:8501](http://localhost:8501)**. Mantenha a janela do terminal aberta enquanto utiliza a aplicação.

> Se `modelo.joblib` ainda não existir (ou se houver diferença de versão das bibliotecas em outro computador), a aplicação treina o modelo automaticamente usando o arquivo local `dataset.csv` (ou baixa a base online caso o CSV não esteja presente).

---

## Como utilizar

1. Preencha as **6 características** do imóvel:
   - Área habitável (m²) — convertida automaticamente para pés quadrados (`ft²`), unidade da base original
   - Quantidade de quartos acima do nível do terreno
   - Quantidade de banheiros completos acima do nível do terreno
   - Ano de construção (`1872` a `2010`)
   - Qualidade geral do imóvel (`1` a `10`)
   - Vagas na garagem
2. Informe o **preço anunciado (US$)** para comparação.
3. Clique em **Estimar preço** para visualizar o valor previsto pela IA, a diferença em dólares e em percentual, além do gráfico comparativo.
4. Abra **Sobre o modelo e resultados** para consultar as métricas de validação e teste.

**Exemplo demonstrado:** `150 m²`, `3 quartos`, `2 banheiros`, `ano 2000`, `qualidade 7`, `2 vagas` e anúncio de `US$ 250.000` → estimativa de **US$ 208.068** (preço anunciado **20,2% acima** da estimativa da IA).

---

## Metodologia de Inteligência Artificial

- **Tipo de problema:** aprendizado supervisionado por **regressão** (variável alvo: `SalePrice`).
- **Pré-processamento:** imputação de valores ausentes pela **mediana** dentro de um `Pipeline` (`SimpleImputer`), evitando vazamento de dados (*data leakage*).
- **Divisão da base:** **60%** treino, **20%** validação e **20%** teste (`random_state=42`).
- **Seleção do modelo:** comparação entre **Regressão Linear** e **Random Forest** (`n_estimators=120`, `min_samples_leaf=2`) pelo **MAE de validação**. O modelo vencedor é reajustado com `treino + validação` (80%) e avaliado nos 20% de **teste** separados.
- O preço anunciado informado pelo usuário **não entra** no treinamento nem na inferência — serve apenas para comparação visual e percentual.

### Resultados obtidos

| Indicador | Resultado |
| :--- | ---: |
| Registros na base (*Ames Housing*) | 2.930 |
| MAE de validação — Regressão Linear | US$ 24.106,38 |
| MAE de validação — Random Forest | **US$ 19.647,16** |
| **MAE de teste (Random Forest)** | **US$ 20.740,00** |
| **RMSE de teste (Random Forest)** | **US$ 33.267,74** |
| **R² de teste (Random Forest)** | **0,862** |

O algoritmo **Random Forest** apresentou erro médio de validação ~18,5% menor que a Regressão Linear e explica cerca de **86,2% da variabilidade** dos preços no conjunto de teste (*R² não representa porcentagem de acurácia*).

---

## Estrutura do projeto

```text
imovel-ia/
├── .streamlit/
│   └── config.toml                # Tema escuro da interface Streamlit
├── docs/
│   └── documentacao.pdf           # Documentação técnica acadêmica
├── Apresentacao_Imovel_IA.pptx    # Slides da apresentação
├── README.md                      # Documentação principal do projeto
├── app.py                         # Interface web interativa em Streamlit
├── dataset.csv                    # Base Ames Housing (2.930 registros)
├── iniciar.bat                    # Script de instalação e execução no Windows
├── metricas.json                  # Métricas geradas no treinamento/teste
├── requirements.txt               # Dependências Python do projeto
├── treinamento.ipynb              # Notebook Jupyter com código, métricas e gráfico
└── treinamento.py                 # Pipeline de treinamento, seleção e avaliação
```

*(O arquivo `modelo.joblib` é gerado automaticamente pelo `treinamento.py` e mantido fora do repositório Git pelo `.gitignore` devido ao tamanho do binário).*

---

## Tecnologias e fontes

- **Linguagem e bibliotecas:** Python, Pandas, NumPy, Scikit-learn, Joblib, Streamlit, Altair e Matplotlib.
- **Base de dados original:** [Ames Housing — Dean De Cock (2011)](https://github.com/topepo/AmesHousing)
- **CSV utilizado:** [AmesHousing.csv](https://github.com/dankicode/Ames_Housing/blob/main/AmesHousing.csv)
- **Referência Kaggle:** [Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset/data)
- **Material de apoio:** [Dive into Deep Learning — House Prices](https://pt.d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html)

---

## Limitações

Os dados referem-se a transações históricas em **Ames, Iowa (EUA), entre 2006 e 2010**, em dólares americanos. Os valores não refletem o mercado atual nem imóveis brasileiros. Área habitável, quartos e banheiros consideram apenas os pavimentos acima do nível do terreno (conforme a base original). Trata-se de uma estimativa acadêmica que **não substitui avaliação imobiliária profissional**.
