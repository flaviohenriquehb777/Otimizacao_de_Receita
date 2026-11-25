<h1 align="center">Otimização de Receita com Machine Learning</h1>

<div align="center">

[![CI](https://github.com/flaviohenriquehb777/Otimizacao_de_Receita/actions/workflows/ci.yml/badge.svg)](https://github.com/flaviohenriquehb777/Otimizacao_de_Receita/actions/workflows/ci.yml)
[![Pages](https://img.shields.io/github/deployments/flaviohenriquehb777/Otimizacao_de_Receita/github-pages?label=pages)](https://github.com/flaviohenriquehb777/Otimizacao_de_Receita/deployments/activity_log?environment=github-pages)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?logo=streamlit)](https://otimizacao-receita.streamlit.app/)
[![DVC](https://img.shields.io/badge/DVC-versionado-9400D3.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-DagsHub-0194E2.svg)](https://dagshub.com/flaviohenriquehb777/Otimizacao_de_Receita)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

**Modelo prescritivo de preço e desconto que maximiza o lucro esperado por linha (SKU × dia),
identificando a combinação ideal de preço final e desconto.**

</div>

<p align="center">
  <a href="https://otimizacao-receita.streamlit.app/" title="Abrir a aplicação de decisão">
    <img src="docs/assets/dashboard_amor_a_cakes.png" alt="Dashboard do App de Decisão" style="width: 720px; max-width: 100%; height: auto;" />
  </a>
</p>

<p align="center">
  <em>Clique na imagem para abrir o <strong>app de decisão</strong> no Streamlit Cloud.</em>
</p>

## Sumário

- [Visão geral do modelo](#visão-geral-do-modelo)
- [Objetivos da análise](#objetivos-da-análise)
- [Estrutura do modelo e do repositório](#estrutura-do-modelo-e-do-repositório)
- [Base de dados](#base-de-dados)
- [Metodologia de análise: CRISP-DM](#metodologia-de-análise-crisp-dm)
- [Resultados-chave e apresentação](#resultados-chave-e-apresentação)
- [Tecnologias utilizadas](#tecnologias-utilizadas)
- [Instalação e uso](#instalação-e-uso)
- [Pipeline reproduzível com DVC](#pipeline-reproduzível-com-dvc)
- [Uso do app](#uso-do-app)
- [Publicação no GitHub Pages](#publicação-no-github-pages)
- [Licença](#licença)
- [Contato](#contato)

---

## Visão geral do modelo

O objetivo **não é "prever vendas"**, e sim **decidir o desconto** que maximiza o lucro.
O projeto separa duas peças:

1. **Modelo de demanda** — aprende `quantidade_vendida_dia` a partir das variáveis de decisão
   (preço/desconto) e do contexto (loja, produto, clima, calendário). É o preditor.
2. **Otimizador de preço** — usa o modelo de demanda para escolher, por linha, o desconto de
   uma grade (0–12%) que maximiza o lucro esperado `(preço_final − custo) × demanda_prevista`,
   respeitando o piso `preco_minimo` por item.

O campeão selecionado por lucro esperado em validação é o **RidgeCV** (regressão linear
regularizada com validação cruzada sobre `alphas = logspace(-3, 3, 13)`). A escolha do modelo
linear é deliberada: garante interpretabilidade e a monotonicidade da curva de demanda.

> **Contexto temporal**: o modelo foi conduzido entre janeiro e junho de 2022. Os artefatos,
> resultados e registros associados refletem esse período e devem ser interpretados nesse
> contexto. A tag git `snapshot` aponta para o commit da versão entregue originalmente.

## Objetivos da análise

- Quantificar a elasticidade da demanda em relação a preço e desconto por SKU.
- Construir um modelo de demanda que **não vaze** informação do futuro (split temporal estrito).
- Selecionar o modelo por **lucro esperado** em validação (não por RMSE isolado).
- Determinar, por linha, o desconto ótimo que maximiza o lucro, respeitando o piso de preço.
- Disponibilizar o resultado em um **app de decisão** que consome o modelo e a lógica de
  negócio reais (sem valores fixos ou hardcoded).

## Estrutura do modelo e do repositório

```text
Otimizacao_de_Receita/
├── app/                         # App de decisão (Streamlit, tema Dark)
│   ├── streamlit_app.py         #   entrypoint (UI fina sobre o simulador)
│   ├── simulador.py             #   lógica de preço/desconto (testável, sem Streamlit)
│   ├── artifacts/               #   bundle de serving (versionado no Git)
│   └── requirements.txt         #   runtime do app (Streamlit Cloud)
├── dados/                       # Dados canônicos em Parquet (governados pelo DVC)
│   ├── dataset_cafeterias_rj.parquet             # base bruta calibrada
│   └── dataset_cafeterias_rj_preparado.parquet   # features + split temporal
├── docs/                        # Site estático (GitHub Pages) + assets
│   ├── index.html               #   showcase (cartão, métricas, curva de negócio)
│   ├── assets/                  #   imagens do README/showcase
│   └── *.json / *.csv           #   artefatos públicos publicados pelo treino
├── notebooks/
│   ├── clean/                   # sem saídas (leitura/execução)
│   └── executed/                # com saídas (espelhos executados)
├── src/
│   ├── config/paths.py          # fonte única de verdade de caminhos
│   ├── data/gerar_dataset.py    # gera a base bruta (determinística, SEED=42)
│   ├── data/preparar_dataset.py # validação fail-closed, features leakage-safe, split
│   └── modeling/train_pipeline.py # treino, seleção e publicação de artefatos
├── models/                      # artefatos do treino (governados pelo DVC)
├── tests/                       # testes unitários e de integração
├── .github/workflows/           # CI (testes/Pages) e MLOps (repro+publicação)
├── dvc.yaml / dvc.lock          # pipeline e travas (gerar→preparar→train)
├── pyproject.toml / uv.lock     # ambiente reprodutível com UV
└── README.md
```

**Princípio de governança**: dados e artefatos pesados (`dados/*.parquet`, `models/*`) **nunca**
são versionados no Git — ficam sob o DVC, com o cache remoto no DagsHub.

## Base de dados

Os dados transacionais reais (POS/ERP, iFood, WhatsApp) não são públicos. Para tornar o projeto
reprodutível de ponta a ponta, a base é **sintética e calibrada** por fatos verificados do
negócio (tabela de preços real, mecânicas promocionais e sazonalidade), com nomenclatura
**de-identificada**. A base canônica é o Parquet `dados/dataset_cafeterias_rj.parquet`
(grão: data × loja × produto), versionado via **DVC**.

**Colunas principais:**

<table align="center">
  <thead>
    <tr>
      <th align="center">Grupo</th>
      <th align="center">Colunas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Preço</td>
      <td align="center"><code>preco_original</code>, <code>preco_final</code>, <code>desconto_pct</code> (0–0,6)</td>
    </tr>
    <tr>
      <td align="center">Contexto econômico</td>
      <td align="center"><code>custo_producao</code>, <code>preco_competidor</code>, <code>indice_preco_competidor</code>, <code>gasto_marketing</code></td>
    </tr>
    <tr>
      <td align="center">Alvo</td>
      <td align="center"><code>quantidade_vendida_dia</code> (transacional) — <code>quantidade_vendida_mes</code> é <strong>agregado</strong> e tratado como vazamento</td>
    </tr>
    <tr>
      <td align="center">Calendário / clima</td>
      <td align="center"><code>feriado_flag</code>, <code>evento_especial</code>, <code>promocao_tipo</code>, <code>mes</code>, <code>dia_semana_idx</code>, <code>temperatura</code>, <code>chuva_mm</code></td>
    </tr>
  </tbody>
</table>

A etapa de *Data Preparation* valida a consistência aritmética
(`preco_final == preco_original × (1 − desconto)`), deriva features de negócio e materializa o
**split temporal estrito**:

<table align="center">
  <thead>
    <tr>
      <th align="center">Split</th>
      <th align="center">Período</th>
      <th align="center">Linhas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Treino</td>
      <td align="center">2024-01-01 → 2025-06-30</td>
      <td align="center">26.256</td>
    </tr>
    <tr>
      <td align="center">Validação</td>
      <td align="center">2025-07-01 → 2025-09-30</td>
      <td align="center">4.416</td>
    </tr>
    <tr>
      <td align="center">Teste (Q4, pico festivo)</td>
      <td align="center">2025-10-01 → 2025-12-31</td>
      <td align="center">4.416</td>
    </tr>
  </tbody>
</table>

## Metodologia de análise: CRISP-DM

O projeto segue o **CRISP-DM**, com cada etapa materializada em código testável e espelhada
nos notebooks:

1. **Business Understanding** — objetivo, KPIs e critérios de sucesso
   (`notebooks/clean/01_business_understanding.ipynb`).
2. **Data Understanding** — esquema, regras de negócio e distribuições
   (`02_data_understanding.ipynb`).
3. **Data Preparation** — validação fail-closed, features **sem vazamento** e split temporal
   estrito (`03_data_preparation.ipynb`, `src/data/preparar_dataset.py`).
4. **Modeling** — treino e seleção do melhor modelo por lucro esperado em holdout
   (`04_modeling.ipynb`, `src/modeling/train_pipeline.py`). *Guardrail* anti-leakage: colunas
   agregadas do próprio dia-alvo (`quantidade_vendida_mes`, `estoque_disponivel`,
   `custo_marketing_unitario`, `ruptura_flag`) nunca entram como preditoras.
5. **Evaluation** — métricas no holdout temporal (Q4) e curva de negócio; o recorte de teste
   é sempre o futuro e **nunca** usado na seleção (`05_evaluation.ipynb`).
6. **Deployment** — os artefatos versionados (base preparada + melhor modelo) alimentam o app;
   o resultado exibido deriva do **modelo real** e da **lógica de negócio real**.

**Honestidade das métricas**: as métricas de demanda (RMSE/MAE/R²) no teste são *factuais*
(comparamos a previsão com a demanda observada). O lucro da política é *contrafactual* —
estimamos a demanda sob descontos que não ocorreram —, logo é uma **estimativa**, reportada com
um baseline ingênuo para calibrar a expectativa.

## Resultados-chave e apresentação

**Candidatos avaliados** (todos com a mesma base de treino e validação):

<table align="center">
  <thead>
    <tr>
      <th align="center">Modelo</th>
      <th align="center">RMSE (val)</th>
      <th align="center">R² (val)</th>
      <th align="center">Uplift de lucro (val)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>RidgeCV</strong> 🏆</td>
      <td align="center">0,7875</td>
      <td align="center">0,8853</td>
      <td align="center"><strong>+4,43%</strong></td>
    </tr>
    <tr>
      <td align="center">RandomForest</td>
      <td align="center">0,6415</td>
      <td align="center">0,9239</td>
      <td align="center">+3,78%</td>
    </tr>
    <tr>
      <td align="center">HistGradientBoosting</td>
      <td align="center">0,6139</td>
      <td align="center">0,9303</td>
      <td align="center">+1,64%</td>
    </tr>
    <tr>
      <td align="center">XGBoostPoisson</td>
      <td align="center">0,6310</td>
      <td align="center">0,9264</td>
      <td align="center">+4,37%</td>
    </tr>
  </tbody>
</table>

O **RidgeCV** foi o campeão por **lucro esperado em validação** (critério de negócio), não por
RMSE — apesar de os modelos de árvore terem menor erro preditivo, eles entregam menos uplift de
lucro na política de desconto.

**Métricas no teste (Q4/2025 — pico festivo, dado nunca visto):**

<table align="center">
  <thead>
    <tr>
      <th align="center">Métrica</th>
      <th align="center">Valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">RMSE</td>
      <td align="center">0,8439</td>
    </tr>
    <tr>
      <td align="center">MAE</td>
      <td align="center">0,4895 (26,2% da média do alvo)</td>
    </tr>
    <tr>
      <td align="center">R²</td>
      <td align="center">0,8966</td>
    </tr>
    <tr>
      <td align="center">Lucro baseline (preço cheio)</td>
      <td align="center">R$ 156,43</td>
    </tr>
    <tr>
      <td align="center">Lucro da política ótima</td>
      <td align="center">R$ 162,12</td>
    </tr>
    <tr>
      <td align="center"><strong>Uplift de lucro</strong></td>
      <td align="center"><strong>+3,64%</strong></td>
    </tr>
    <tr>
      <td align="center">Desconto médio aplicado</td>
      <td align="center">5,1%</td>
    </tr>
    <tr>
      <td align="center">Linhas com desconto &gt; 0</td>
      <td align="center">52,9%</td>
    </tr>
  </tbody>
</table>

**Feature importance** (permutation importance, top 5): `promocao_tipo` (23,6), `produto` (6,4),
`tamanho_embalagem` (3,7), `preco_original` (3,6), `tipo_produto` (3,4).

> Todos os números acima são extraídos diretamente de `docs/metrics_snapshot.json` e
> `docs/model_card.json`, publicados pelo próprio pipeline de treino.

## Tecnologias utilizadas

<table align="center">
  <thead>
    <tr>
      <th align="center">Categoria</th>
      <th align="center">Tecnologias</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Linguagem / ambiente</td>
      <td align="center"><strong>Python 3.12</strong>, <strong>UV</strong> (<code>pyproject.toml</code> + <code>uv.lock</code>)</td>
    </tr>
    <tr>
      <td align="center">Dados</td>
      <td align="center"><strong>Pandas</strong>, <strong>NumPy</strong>, <strong>Parquet</strong> (PyArrow)</td>
    </tr>
    <tr>
      <td align="center">Modelagem</td>
      <td align="center"><strong>scikit-learn</strong> (RidgeCV, Pipeline, ColumnTransformer, permutation importance), <strong>XGBoost</strong> (candidato Poisson)</td>
    </tr>
    <tr>
      <td align="center">Experimentação</td>
      <td align="center"><strong>MLflow</strong> (tracking), <strong>DagsHub</strong> (remote de experimentos + cache DVC)</td>
    </tr>
    <tr>
      <td align="center">Versionamento de dados</td>
      <td align="center"><strong>DVC</strong> (<code>dvc.yaml</code> + <code>dvc.lock</code>)</td>
    </tr>
    <tr>
      <td align="center">Aplicação</td>
      <td align="center"><strong>Streamlit</strong> (app de decisão, tema Dark)</td>
    </tr>
    <tr>
      <td align="center">Visualização</td>
      <td align="center"><strong>Altair</strong> (app), <strong>Chart.js</strong> (showcase), <strong>Matplotlib</strong>, <strong>Seaborn</strong></td>
    </tr>
    <tr>
      <td align="center">Qualidade</td>
      <td align="center"><strong>Ruff</strong>, <strong>Black</strong>, <strong>pytest</strong> (23 testes)</td>
    </tr>
    <tr>
      <td align="center">CI/CD</td>
      <td align="center"><strong>GitHub Actions</strong> (Node 24), <strong>GitHub Pages</strong></td>
    </tr>
  </tbody>
</table>

## Instalação e uso

**Pré-requisitos:** Python 3.12 e [UV](https://docs.astral.sh/uv/).

```bash
# 1. Clone o repositório
git clone https://github.com/flaviohenriquehb777/Otimizacao_de_Receita.git
cd Otimizacao_de_Receita

# 2. Instale as dependências (todos os grupos: dev + test + mlops)
uv sync --all-groups

# 3. Reproduza o pipeline completo (dados → modelo)
uv run dvc repro

# 4. Execute os testes
uv run pytest -q

# 5. (Opcional) Explore os notebooks CRISP-DM
uv run jupyter lab
# Abra notebooks/clean/ (sem saídas) ou notebooks/executed/ (com saídas)
```

## Pipeline reproduzível com DVC

O pipeline é definido em [`dvc.yaml`](dvc.yaml) e travado em [`dvc.lock`](dvc.lock):

```text
gerar_dados  →  preparar_dados  →  train
   Parquet        Parquet          best_model.pkl
  (bruto)      (preparado)         metrics_snapshot.json
                                    model_card.json
                                    business_curve.csv
                                    test_predictions.parquet
```

```bash
# Visualizar o DAG
uv run dvc dag

# Reproduzir (só roda o que mudou)
uv run dvc repro

# Verificar status vs. lock
uv run dvc status

# Puxar artefatos do cache remoto (DagsHub)
uv run dvc pull
```

O remote do DVC é o DagsHub (`https://dagshub.com/flaviohenriquehb777/Otimizacao_de_Receita.dvc`);
as credenciais ficam em `.dvc/config.local` (gitignored). **A CI não publica no DagsHub** —
isso acontece sob demanda, localmente, para evitar expor segredos nos workflows.

## Uso do app

O **app de decisão** consome o bundle de serving em `app/artifacts/` (versionado no Git) —
não precisa de DVC nem da base crua, e **nenhum secret** é necessário.

**Localmente:**

```bash
uv run streamlit run app/streamlit_app.py
```

**Na nuvem — Streamlit Cloud:** já publicado em
[**otimizacao-receita.streamlit.app**](https://otimizacao-receita.streamlit.app/).

O app permite escolher produto, loja, mês, clima e contexto promocional, e retorna:
- o **desconto ótimo** e o **preço final** que maximizam o lucro esperado;
- a **curva de demanda/receita/lucro** por ponto da grade de desconto (0–12%);
- o **uplift** sobre o preço cheio (baseline).

## Publicação no GitHub Pages

O **showcase estático** em [`docs/index.html`](docs/index.html) consome os artefatos
publicados pelo treino (`metrics_snapshot.json`, `model_card.json`, `business_curve.csv`) e
exibe o cartão do modelo, as métricas e a curva de negócio. A CI publica o `docs/` no Pages a
cada push na `main`.

🔗 **Showcase**: https://flaviohenriquehb777.github.io/Otimizacao_de_Receita/

O botão do app no showcase aponta para o Streamlit Cloud.

## Licença

Este projeto está licenciado sob a **Licença MIT**. Consulte [LICENSE.md](LICENSE.md) para mais
detalhes.

## Contato

<table align="center">
  <tr>
    <th align="center">Nome</th>
    <th align="center">LinkedIn</th>
    <th align="center">E-mail</th>
    <th align="center">GitHub</th>
  </tr>
  <tr>
    <td align="center">Flávio Henrique Barbosa</td>
    <td align="center"><a href="https://www.linkedin.com/in/fl%C3%A1vio-henrique-barbosa-38465938">Flávio Henrique Barbosa</a></td>
    <td align="center"><a href="mailto:flaviohenriquehb777@outlook.com">flaviohenriquehb777@outlook.com</a></td>
    <td align="center"><a href="https://github.com/flaviohenriquehb777">@flaviohenriquehb777</a></td>
  </tr>
</table>
