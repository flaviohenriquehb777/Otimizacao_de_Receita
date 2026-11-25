<h1 align="center">Otimização de Receita com Machine Learning</h1>

[![Build CI](https://github.com/flaviohenriquehb777/Otimizacao_de_Receita/actions/workflows/ci.yml/badge.svg)](https://github.com/flaviohenriquehb777/Otimizacao_de_Receita/actions/workflows/ci.yml)
[![Pages](https://img.shields.io/github/deployments/flaviohenriquehb777/Otimizacao_de_Receita/github-pages?label=pages)](https://github.com/flaviohenriquehb777/Otimizacao_de_Receita/deployments/activity_log?environment=github-pages)

Clique na miniatura para abrir a aplicação:

<p align="center">
  <a href="https://flaviohenriquehb777.github.io/Otimizacao_de_Receita/index.html?nocache=20220630" title="Abrir aplicação">
    <img src="docs/thumbnail_card.svg" alt="Miniatura da Aplicação" style="width: 640px; max-width: 100%; height: auto;" />
  </a>
  
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

**Modelo de análise de dados de vendas e modelagem preditiva para otimização da receita, identificando a combinação ideal de preço de venda e desconto para maximizar lucros.**

## Sumário
- [Visão Geral do Modelo](#visão-geral-do-modelo)
- [Objetivos](#objetivos)
- [Estrutura do Modelo](#estrutura-do-modelo)
- [Base de Dados](#base-de-dados)
- [Metodologia de Análise e Modelagem](#metodologia-de-análise-e-modelagem)
- [Resultados Chave e Recomendações](#resultados-chave-e-recomendações)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação e Uso](#instalação-e-uso)
- [Licença](#licença)
- [Contato](#contato)

## Contexto Temporal do Modelo

O modelo foi conduzido entre janeiro e junho de 2022. Os artefatos publicados, os resultados e os registros associados refletem esse período e devem ser interpretados nesse contexto temporal.

Para deixar explícito o estado da entrega original, a tag git `snapshot` aponta para o commit que representa a versão entregue.

## Visão Geral do Modelo:

Este modelo foca na aplicação de Machine Learning (Regressão Linear) para analisar dados de vendas e determinar as estratégias ideais de precificação e desconto que maximizem a receita. Através de uma análise exploratória detalhada e da construção de um modelo preditivo, buscamos fornecer insights acionáveis para otimização do desempenho de vendas.

## Objetivos:

* Realizar uma análise exploratória completa da base de dados de vendas.
* Entender a relação entre 'PrecoVenda', 'PrecoOriginal', 'Desconto' e 'VendaQtd'.
* Construir e avaliar um modelo de regressão capaz de prever a quantidade vendida.
* Identificar a combinação ideal de preço de venda e desconto para maximizar a receita.

## Estrutura do Modelo

- `dados/`: dados canônicos em **Parquet**, governados pelo **DVC** (nunca versionados no Git).
  - `dataset_cafeterias_rj.parquet`: base bruta calibrada (saída de `gerar_dados`).
  - `dataset_cafeterias_rj_preparado.parquet`: base preparada, com features e split temporal (saída de `preparar_dados`).
- `notebooks/`: cadernos Jupyter organizados por CRISP-DM.
  - `clean/`: sem saídas, prontos para leitura e execução.
    - `01_business_understanding.ipynb`, `02_data_understanding.ipynb`, `03_data_preparation.ipynb`
  - `executed/`: cópias executadas (com saídas e `execution_count`).
- `src/`: código-fonte
  - `config/paths.py`: fonte única de verdade de caminhos e constantes
  - `data/gerar_dataset.py`: gera a base bruta calibrada (determinística, `SEED = 42`)
  - `data/preparar_dataset.py`: validação fail-closed, limpeza, features leakage-safe e split temporal
  - `modeling/train_pipeline.py`: treino, seleção do melhor modelo e publicação de artefatos
- `models/`: artefatos do treino, governados pelo **DVC** (fora do Git).
- `app/`: app de decisão (**Streamlit**, tema Dark) e o bundle de serving enxuto em `app/artifacts/` (versionado no Git).
- `dvc.yaml` / `dvc.lock`: definição e travas do pipeline (`gerar_dados → preparar_dados → train`).
- `docs/`: site estático (GitHub Pages) que consome os artefatos publicados pelo treino.
- `.github/workflows/`: CI (testes/qualidade) e MLOps (pipeline + publicação).
- `tests/`: testes unitários e de integração.
- `pyproject.toml` / `uv.lock`: ambiente reprodutível com **UV**.

## Base de Dados:

Os dados transacionais reais (POS/ERP, iFood, WhatsApp) não são públicos. Para tornar o projeto
reprodutível de ponta a ponta, a base é **sintética e calibrada** por fatos verificados do negócio
(tabela de preços real, mecânicas promocionais e sazonalidade), com nomenclatura
**de-identificada**. A base canônica é o Parquet `dados/dataset_cafeterias_rj.parquet`
(grão: data × loja × produto), versionado via **DVC**.

Colunas principais:

* `preco_original` / `preco_final` / `desconto_pct`: preço cheio, preço praticado e desconto (0–0,6).
* `custo_producao`, `preco_competidor`, `gasto_marketing`.
* `quantidade_vendida_dia` (alvo transacional) e `quantidade_vendida_mes` (**agregado** — tratado como vazamento na modelagem).
* Sazonalidade/calendário: `feriado_flag`, `evento_especial`, `promocao_tipo`.

A etapa de *Data Preparation* valida a consistência aritmética
(`preco_final == preco_original × (1 − desconto)`), deriva features de negócio e materializa o
**split temporal estrito** (treino ≤ 2025-06-30; validação ≤ 2025-09-30; teste = Q4/2025).

## Metodologia de Análise e Modelagem

O projeto segue o **CRISP-DM**, com cada etapa materializada em código testável e espelhada nos notebooks:

1. **Business Understanding** — objetivo, KPIs e critérios de sucesso (`notebooks/clean/01_business_understanding.ipynb`).
2. **Data Understanding** — esquema, regras de negócio e distribuições (`02_data_understanding.ipynb`).
3. **Data Preparation** — validação fail-closed, features **sem vazamento** e split temporal estrito (`03_data_preparation.ipynb`, `src/data/preparar_dataset.py`).
4. **Modeling** — treino e seleção do melhor modelo por lucro esperado em holdout (`src/modeling/train_pipeline.py`). *Guardrail* anti-leakage: colunas agregadas do próprio dia-alvo nunca entram como preditoras.
5. **Evaluation** — métricas no holdout temporal (Q4) e curva de negócio; o recorte de teste é sempre o futuro.
6. **Deployment** — os artefatos versionados (base preparada + melhor modelo) alimentam o app; o resultado exibido deriva do **modelo real** e da **lógica de negócio real**, nunca de valores fixos.

## Resultados Chave e Recomendações:

Os resultados dependem dos coeficientes reais treinados no seu ambiente. A aplicação exibirá o desconto ideal, o preço final e a receita estimada com base no **modelo exportado**. Recomenda-se reavaliar periodicamente e monitorar a elasticidade de demanda em campanhas reais.

## Tecnologias Utilizadas:

* **Python:** Linguagem de programação principal.
* **Pandas:** Para manipulação e análise de dados tabulares.
* **NumPy:** Para operações numéricas de alto desempenho.
* **Scikit-learn (sklearn):** Para pré-processamento e modelagem (LinearRegression, Pipeline), divisão de dados (train_test_split) e avaliação de modelos (RMSE, MAE, R²).
* **Matplotlib:** Para criação de gráficos, especialmente a curva de aprendizagem.
* **Seaborn:** Para visualizações estatísticas e aprimoramento estético dos gráficos.
* **XGBoost:** Como candidato não linear com restrições condicionais ao alvo.
* **Streamlit + Altair:** App de decisão (tema Dark) que consome o modelo real e a lógica de negócio real.

## Instalação e Uso

1. **Pré-requisitos**
   - Python 3.12
   - [UV](https://docs.astral.sh/uv/) — gerencia `pyproject.toml` + `uv.lock`

2. **Clone o repositório**
    ```bash
    git clone https://github.com/flaviohenriquehb777/Otimizacao_de_Receita.git
    cd Otimizacao_de_Receita
    ```

3. **Instale as dependências**
    ```bash
    uv sync
    ```

4. **Reproduza o pipeline completo (dados → modelo)**
    ```bash
    uv run dvc repro
    ```
    O DAG é `gerar_dados → preparar_dados → train`, e o `dvc.lock` trava a reprodutibilidade.
    Para inspecionar: `uv run dvc dag`.

5. **Execute os testes**
    ```bash
    uv run pytest -q
    ```

6. **(Opcional) Explore os notebooks CRISP-DM**
    ```bash
    uv run jupyter lab
    ```
    Abra `notebooks/clean/` (sem saídas) ou `notebooks/executed/` (com saídas). Os espelhos
    executados são gerados por `scripts/exec_notebook.py`.

7. **Suba o site estático localmente**
    ```bash
    uv run python -m http.server 8000
    # Abra http://localhost:8000/docs/
    ```

## Boas práticas e versão de arquivos

- Dados e artefatos de modelo **não** são versionados no Git: a governança é do **DVC** (`dvc.yaml` + `dvc.lock`).
- Tudo em `src/` é determinístico (`SEED = 42`) e coberto por testes; os notebooks apenas orquestram o código.
- A mesma lógica de preparação usada no pipeline é reaproveitada pelos notebooks e pelo app — sem duplicação.
- A CI instala dependências, roda `pytest` e valida o pipeline; o workflow de MLOps reproduz e publica os artefatos.

## Showcase (GitHub Pages) e App de Decisão

- O **app de decisão** (`app/streamlit_app.py`, tema Dark) consome o **modelo real** e a **lógica de negócio real** — os mesmos `aplicar_desconto`/otimizador do pipeline, sem duplicação. Ele lê o bundle de serving enxuto em `app/artifacts/` (sem DVC nem a base crua).
- O **showcase** em `docs/index.html` (GitHub Pages) é uma página estática de leitura: exibe o cartão de modelo, o snapshot de métricas e a curva de negócio publicados pelo treino, além do link para o app. Não há inferência no navegador — a interatividade fica no app.
- O pipeline publica dois destinos: o bundle de serving em `app/artifacts/` (`best_model.pkl`, `model_card.json`, `metrics_snapshot.json`, `business_curve.csv`) e os artefatos públicos do showcase em `docs/` (`metrics_snapshot.json`, `model_card.json`, `business_curve.csv`).
- A URL do app no showcase é definida pela constante `APP_URL` no topo do script de `docs/index.html` (preenchida no deploy). Enquanto vazia, o botão do app fica oculto e restam os links do repositório.

### Deploy do app (Streamlit Cloud)

O app é **auto-contido**: lê apenas o bundle versionado em `app/artifacts/` (sem DVC nem a base crua) e nenhum secret é necessário.

1. Em https://share.streamlit.io, escolha **New app** apontando para o repositório `flaviohenriquehb777/Otimizacao_de_Receita`, branch `main`, arquivo principal `app/streamlit_app.py`.
2. Aguarde o build; o tema Dark já vem de `.streamlit/config.toml`.
3. Copie a URL pública e preencha a constante `APP_URL` no `docs/index.html` para habilitar o CTA do app no showcase.

## Integração com DagsHub (MLflow)

O treino registra **um run por candidato** no MLflow e marca o campeão (`model_best`).
Por padrão o tracking é **local** (`mlruns/`); para publicar no DagsHub, exporte
**apenas** as variáveis namespaced deste projeto (fail-closed — as genéricas
`DAGSHUB_TOKEN` / `MLFLOW_TRACKING_*` são deliberadamente ignoradas para não herdar
credenciais de outros projetos):

```bash
export OTIMIZACAO_DAGSHUB_OWNER="<owner>"
export OTIMIZACAO_DAGSHUB_REPO="Otimizacao_de_Receita"
export OTIMIZACAO_MLFLOW_TRACKING_USERNAME="<owner>"
export OTIMIZACAO_MLFLOW_TRACKING_PASSWORD="<token>"
uv run python -m src.modeling.train_pipeline
```

- Repositório no DagsHub (experimentos): https://dagshub.com/flaviohenriquehb777/Otimizacao_de_Receita
- O token também pode vir por `OTIMIZACAO_MLFLOW_TRACKING_TOKEN` (ou `OTIMIZACAO_DAGSHUB_TOKEN`).
- Sem credenciais, o pipeline segue com tracking local, sem falhar.
- A **CI não publica** no DagsHub: os experimentos são registrados localmente/sob demanda,
  evitando expor segredos nos workflows.

## Licença:

Este modelo está licenciado sob a Licença MIT. Para mais detalhes, consulte o arquivo [LICENSE.md](LICENSE.md) na raiz do repositório.

## Contato:

Se tiver alguma dúvida, sugestão ou quiser colaborar, sinta-se à vontade para entrar em contato:
-   **Nome:** Flávio Henrique Barbosa
-   **LinkedIn:** [Flávio Henrique Barbosa | LinkedIn](https://www.linkedin.com/in/fl%C3%A1vio-henrique-barbosa-38465938)
-   **Email:** flaviohenriquehb777@outlook.com
