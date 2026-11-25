"""Modelagem prescritiva de preço e desconto (CRISP-DM — Modeling / Evaluation).

Framing de negócio
------------------
O objetivo não é "prever vendas", é **decidir o desconto** que maximiza o lucro.
Para isso separamos duas coisas:

1. **Modelo de demanda** — aprende `quantidade_vendida_dia` a partir das
   variáveis de decisão (preço/desconto) e do contexto (loja, produto, clima,
   calendário). É o preditor.
2. **Otimizador de preço** — usa o modelo de demanda para escolher, por linha, o
   desconto de uma grade que maximiza o lucro esperado
   `(preço_final − custo) × demanda_prevista`, respeitando o piso de preço
   (`preco_minimo`).

Protocolo temporal honesto (anti-leakage)
-----------------------------------------
A base preparada já traz o split temporal estrito. A avaliação é sempre
passado → futuro, sem embaralhar:

* **treino** (≤ 2025-06-30): ajuste dos candidatos;
* **validação** (jul–set/2025): seleção do modelo por lucro esperado;
* **teste** (Q4/2025, o pico festivo): relatório final, nunca usado na seleção.

O campeão é **re-ajustado em treino + validação** antes do teste, para usar todo
o passado disponível — mantendo o teste intocado.

Honestidade das métricas
------------------------
As métricas de demanda (RMSE/MAE/R²) no teste são *factuais* (comparamos a
previsão com a demanda realmente observada). Já o lucro da política é
*contrafactual*: estimamos a demanda sob descontos que não ocorreram, logo é uma
**estimativa** — reportada como tal, junto de um baseline ingênuo para calibrar a
expectativa.

Reprodutibilidade
-----------------
Determinístico (`SEED = 42`); a seleção usa apenas validação; o teste é o futuro.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.paths import APP_ARTIFACTS_DIR, DADOS_CAFETERIAS_PREPARADO, MODELS_DIR
from src.data.preparar_dataset import colunas_com_vazamento

try:  # XGBoost é opcional para não travar ambientes mínimos.
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - dependência opcional
    XGBRegressor = None  # type: ignore[assignment]

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover - MLflow é opcional

    class _NoopRunInfo:
        run_id = "no-mlflow"

    class _NoopRun:
        info = _NoopRunInfo()

    class _NoopMlflow:
        def set_tracking_uri(self, *args, **kwargs):
            return None

        def set_experiment(self, *args, **kwargs):
            return None

        def start_run(self, *args, **kwargs):
            return contextlib.nullcontext(_NoopRun())

        def log_params(self, *args, **kwargs):
            return None

        def log_metrics(self, *args, **kwargs):
            return None

        def log_artifact(self, *args, **kwargs):
            return None

        def set_tag(self, *args, **kwargs):
            return None

        def get_tracking_uri(self):
            return "local"

    mlflow = _NoopMlflow()

SEED = 42
TARGET = "quantidade_vendida_dia"

# --- Contrato de features (explícito e leakage-safe) -------------------------
# Curado manualmente e validado contra o guardrail de vazamento da camada de
# dados (`colunas_com_vazamento`). Nunca incluir agregados do próprio dia-alvo
# (`quantidade_vendida_mes`) nem desfechos derivados da demanda
# (`estoque_disponivel`, `ruptura_flag`, `custo_marketing_unitario`).
#
# `ano`/`trimestre` são deliberadamente excluídos: com split temporal, o modelo
# não pode "saber" o ano para extrapolar tendência — só a sazonalidade (`mes`).
FEATURES_NUMERICAS: tuple[str, ...] = (
    # Variáveis de decisão (o otimizador manipula estas).
    "preco_original",
    "desconto_pct",
    "preco_final",
    # Estrutura de custo e competição.
    "custo_producao",
    "preco_minimo",
    "preco_competidor",
    "indice_preco_competidor",
    # Contexto de demanda.
    "gasto_marketing",
    "temperatura",
    "chuva_mm",
    "mes",
    "dia_semana_idx",
    "fim_de_semana",
    "feriado_flag",
    "promocao_flag",
)

FEATURES_CATEGORICAS: tuple[str, ...] = (
    "loja",
    "produto",
    "categoria",
    "canal",
    "evento_especial",
    "promocao_tipo",
    "tipo_produto",
    "tamanho_embalagem",
    "perfil_loja",
)

# Grade de descontos avaliada pelo otimizador. Cap em 12% = teto do desconto de
# balcão **sem promoção** (U(0,03; 0,12)), a única região em que o contrafactual
# "desconto sem promoção" está identificado nos dados. Acima disso os descontos
# observados (Item do Mês, até 16,4%) vêm sempre atrelados a um lift promocional,
# e os aprendizes de árvore extrapolam o lucro para cima — auditoria: na grade
# [0; 0,16] o HGB infla +71% e o RF +37% de uplift, contra a verdade estritamente
# decrescente. Limitar a grade mantém a curva de lucro monotônica e o uplift
# defensável. Baseline e verdade concordam que o ótimo está na ponta barata.
GRADE_DESCONTO: np.ndarray = np.round(np.arange(0.0, 0.1201, 0.01), 4)


@dataclass(frozen=True)
class ModelSpec:
    """Candidato de modelo: nome legível + pipeline sklearn pronto para ajuste."""

    nome: str
    pipeline: Pipeline
    params: dict[str, object]


# --- 1) Dados -----------------------------------------------------------------


def carregar_base_preparada(caminho: Path | None = None) -> pd.DataFrame:
    """Carrega a base preparada canônica (features leakage-safe + split)."""
    caminho = caminho or DADOS_CAFETERIAS_PREPARADO
    if not caminho.exists():
        raise FileNotFoundError(
            f"Base preparada não encontrada em {caminho}. "
            "Rode 'dvc repro' (ou 'python -m src.data.preparar_dataset')."
        )
    return pd.read_parquet(caminho)


def validar_contrato_features(df: pd.DataFrame) -> None:
    """Fail-closed: o contrato de features precisa existir e ser leakage-safe."""
    features = (*FEATURES_NUMERICAS, *FEATURES_CATEGORICAS)
    faltantes = [c for c in features if c not in df.columns]
    if faltantes:
        raise ValueError(f"Features ausentes na base preparada: {faltantes}")
    vazamento = sorted(set(features) & set(colunas_com_vazamento()))
    if vazamento:
        raise ValueError(f"Vazamento de dados no contrato de features: {vazamento}")
    if TARGET in features:
        raise ValueError("O alvo não pode ser feature.")


def separar_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Devolve as vistas temporalmente disjuntas treino/validação/teste."""
    splits = {
        nome: df[df["split"] == nome].copy()
        for nome in ("treino", "validacao", "teste")
    }
    vazio = [nome for nome, parte in splits.items() if parte.empty]
    if vazio:
        raise ValueError(f"Splits vazios na base preparada: {vazio}")
    return splits


def _hash_dados(df: pd.DataFrame) -> str:
    """Hash determinístico do conteúdo (proveniência leve dos dados usados)."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


# --- 2) Otimizador de preço ---------------------------------------------------


def aplicar_desconto(X: pd.DataFrame, desconto: float) -> pd.DataFrame:
    """Reescreve o cenário para um desconto `d`, mantendo as features coerentes.

    Público porque é parte do **contrato de serving**: o app reusa exatamente
    esta regra de negócio (sem duplicar a lógica) ao simular um cenário.
    """
    Xd = X.copy()
    preco_final = (X["preco_original"] * (1.0 - desconto)).round(2)
    Xd["desconto_pct"] = desconto
    Xd["preco_final"] = preco_final
    if "indice_preco_competidor" in Xd.columns:
        Xd["indice_preco_competidor"] = (preco_final / X["preco_competidor"]).round(4)
    return Xd


def matriz_lucro(
    modelo, X: pd.DataFrame, grade: np.ndarray = GRADE_DESCONTO
) -> np.ndarray:
    """Lucro esperado por linha e por desconto, respeitando o piso de preço.

    Retorna uma matriz ``(n_linhas, n_descontos)``. Cenários que derrubam o preço
    abaixo de `preco_minimo` recebem ``-inf`` (inviáveis); a coluna sem desconto é
    sempre factível, garantindo ao menos uma opção por linha.
    """
    preco_original = X["preco_original"].to_numpy(dtype=float)
    custo = X["custo_producao"].to_numpy(dtype=float)
    preco_minimo = X["preco_minimo"].to_numpy(dtype=float)

    lucro = np.empty((len(X), len(grade)), dtype=float)
    for j, d in enumerate(grade):
        qtd = np.asarray(modelo.predict(aplicar_desconto(X, float(d))), dtype=float)
        preco = preco_original * (1.0 - float(d))
        lucro[:, j] = (preco - custo) * qtd

    preco_grade = preco_original[:, None] * (1.0 - grade)[None, :]
    inviavel = preco_grade < preco_minimo[:, None]
    inviavel[:, 0] = False
    return np.where(inviavel, -np.inf, lucro)


def politica_otima(
    modelo, X: pd.DataFrame, grade: np.ndarray = GRADE_DESCONTO
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Desconto ótimo por linha, lucro da política e lucro do baseline (preço cheio)."""
    lucro = matriz_lucro(modelo, X, grade)
    melhor = np.argmax(lucro, axis=1)
    linhas = np.arange(len(X))
    return grade[melhor], lucro[linhas, melhor], lucro[:, 0]


def metricas_negocio(modelo, X: pd.DataFrame) -> dict[str, float]:
    """Resume o valor de negócio da política otimizada versus o preço de tabela."""
    desconto, lucro_politica, lucro_baseline = politica_otima(modelo, X)
    baseline = float(np.mean(lucro_baseline))
    politica = float(np.mean(lucro_politica))
    uplift = float((politica - baseline) / baseline * 100.0) if baseline else 0.0
    return {
        "lucro_baseline": baseline,
        "lucro_politica": politica,
        "uplift_pct": uplift,
        "desconto_medio": float(np.mean(desconto)),
        "pct_linhas_com_desconto": float(np.mean(desconto > 0.0)),
    }


# --- 3) Modelos ---------------------------------------------------------------


def _preprocessador() -> ColumnTransformer:
    """Escala numéricas e codifica categóricas (one-hot), sem vazamento de fit."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(FEATURES_NUMERICAS)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(FEATURES_CATEGORICAS)),
        ]
    )


def _largura_onehot(X: pd.DataFrame) -> int:
    """Nº de colunas geradas pelo one-hot (para alinhar restrições monotônicas)."""
    encoder = OneHotEncoder(handle_unknown="ignore").fit(X[list(FEATURES_CATEGORICAS)])
    return int(sum(len(categorias) for categorias in encoder.categories_))


def _restricoes_monotonas(largura_onehot: int) -> tuple[int, ...]:
    """Impõe o prior econômico: demanda cai com o preço e sobe com o desconto.

    O `ColumnTransformer` emite numéricas primeiro; logo o vetor de restrições
    segue a ordem de `FEATURES_NUMERICAS` e o bloco one-hot fica neutro (0).
    """
    sinais = {"preco_final": -1, "desconto_pct": +1}
    return (
        tuple(sinais.get(coluna, 0) for coluna in FEATURES_NUMERICAS)
        + (0,) * largura_onehot
    )


def construir_candidatos(largura_onehot: int) -> list[ModelSpec]:
    """Monta os candidatos: baseline linear, floresta, boosting e XGBoost Poisson."""
    candidatos = [
        ModelSpec(
            "RidgeCV",
            Pipeline(
                [
                    ("prep", _preprocessador()),
                    ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
                ]
            ),
            {"alphas": "logspace(-3,3,13)"},
        ),
        ModelSpec(
            "RandomForest",
            Pipeline(
                [
                    ("prep", _preprocessador()),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300,
                            min_samples_leaf=5,
                            n_jobs=-1,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            {"n_estimators": 300, "min_samples_leaf": 5},
        ),
        ModelSpec(
            "HistGradientBoosting",
            Pipeline(
                [
                    ("prep", _preprocessador()),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=400, learning_rate=0.06, random_state=SEED
                        ),
                    ),
                ]
            ),
            {"max_iter": 400, "learning_rate": 0.06},
        ),
    ]
    if XGBRegressor is not None:
        restricoes = _restricoes_monotonas(largura_onehot)
        candidatos.append(
            ModelSpec(
                "XGBoostPoisson",
                Pipeline(
                    [
                        ("prep", _preprocessador()),
                        (
                            "model",
                            XGBRegressor(
                                objective="count:poisson",
                                n_estimators=600,
                                learning_rate=0.03,
                                max_depth=5,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                reg_lambda=2.0,
                                min_child_weight=5,
                                tree_method="hist",
                                monotone_constraints=restricoes,
                                random_state=SEED,
                            ),
                        ),
                    ]
                ),
                {
                    "objective": "count:poisson",
                    "n_estimators": 600,
                    "learning_rate": 0.03,
                    "monotone_constraints": "preco_final=-1, desconto_pct=+1",
                },
            )
        )
    return candidatos


# --- 4) Avaliação -------------------------------------------------------------


def metricas_demanda(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Métricas factuais de previsão de demanda (+ MAE normalizado pela média)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    media = float(np.mean(y_true))
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": mae,
        "r2": float(r2_score(y_true, y_pred)),
        "mae_ratio": float(mae / media) if media else float("nan"),
    }


def baselines_ingenuos(
    treino: pd.DataFrame, X_eval: pd.DataFrame, y_eval: np.ndarray
) -> dict[str, float]:
    """RMSE de referências sem modelo: média global e média por (loja, produto)."""
    media_global = float(treino[TARGET].mean())
    rmse_global = float(
        root_mean_squared_error(y_eval, np.full(len(y_eval), media_global))
    )

    por_sku = treino.groupby(["loja", "produto"])[TARGET].mean()
    chave = pd.MultiIndex.from_frame(X_eval[["loja", "produto"]])
    pred_sku = por_sku.reindex(chave).fillna(media_global).to_numpy(dtype=float)
    rmse_sku = float(root_mean_squared_error(y_eval, pred_sku))
    return {
        "rmse_media_global": rmse_global,
        "rmse_media_sku": rmse_sku,
        "media_alvo_treino": media_global,
    }


def importancias_por_feature(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    """Importância das features originais (legível para negócio).

    Usa o atributo nativo quando existe — coeficientes (lineares) ou importâncias
    (árvores) — somando o bloco one-hot de volta ao nome da feature original.
    Quando o estimador não expõe nenhum dos dois (ex.: HistGradientBoosting),
    cai em *permutation importance* sobre as features originais, que já é nativa
    por feature e vale para qualquer modelo.
    """
    preprocessador = pipeline.named_steps["prep"]
    modelo = pipeline.named_steps["model"]

    if hasattr(modelo, "feature_importances_"):
        valores = np.asarray(modelo.feature_importances_, dtype=float)
    elif hasattr(modelo, "coef_"):
        valores = np.abs(np.asarray(modelo.coef_, dtype=float)).ravel()
    else:
        perm = permutation_importance(
            pipeline, X, y, n_repeats=8, random_state=SEED, n_jobs=-1
        )
        nomes = (*FEATURES_NUMERICAS, *FEATURES_CATEGORICAS)
        ordenado = sorted(
            zip(nomes, perm.importances_mean, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return {nome: float(valor) for nome, valor in ordenado}

    agregado: dict[str, float] = {}
    cursor = 0
    for coluna in FEATURES_NUMERICAS:
        agregado[coluna] = float(valores[cursor])
        cursor += 1
    encoder = preprocessador.named_transformers_["cat"]
    for coluna, categorias in zip(
        FEATURES_CATEGORICAS, encoder.categories_, strict=True
    ):
        largura = len(categorias)
        agregado[coluna] = float(np.sum(valores[cursor : cursor + largura]))
        cursor += largura
    assert cursor == len(
        valores
    ), "Ordem do preprocessador divergiu do contrato de features."

    ordenado = sorted(agregado.items(), key=lambda item: item[1], reverse=True)
    return dict(ordenado)


# --- 5) Artefatos -------------------------------------------------------------


def _curva_negocio(
    modelo, X: pd.DataFrame, grade: np.ndarray = GRADE_DESCONTO
) -> pd.DataFrame:
    """Curva média prevista (demanda/receita/lucro) por desconto, para o site/app."""
    linhas = []
    for d in grade:
        qtd = np.asarray(modelo.predict(aplicar_desconto(X, float(d))), dtype=float)
        preco = X["preco_original"].to_numpy(dtype=float) * (1.0 - float(d))
        receita = preco * qtd
        lucro = (preco - X["custo_producao"].to_numpy(dtype=float)) * qtd
        linhas.append(
            {
                "desconto_pct": float(d),
                "qtd_media_prevista": float(np.mean(qtd)),
                "receita_media_prevista": float(np.mean(receita)),
                "lucro_medio_previsto": float(np.mean(lucro)),
            }
        )
    return pd.DataFrame(linhas)


def _enriquecer(modelo, X: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Métricas combinadas (demanda + negócio) e a tabela de cenários por linha."""
    y_pred = np.asarray(modelo.predict(X), dtype=float)
    desconto, lucro_politica, lucro_baseline = politica_otima(modelo, X)
    metricas = {
        **metricas_demanda(X[TARGET].to_numpy(dtype=float), y_pred),
        **metricas_negocio(modelo, X),
    }
    cenarios = X[
        [
            "data",
            "loja",
            "produto",
            "preco_original",
            "preco_minimo",
            "custo_producao",
            TARGET,
        ]
    ].copy()
    cenarios["demanda_prevista"] = y_pred
    cenarios["desconto_otimo"] = desconto
    cenarios["preco_otimo"] = (cenarios["preco_original"] * (1.0 - desconto)).round(2)
    cenarios["lucro_politica"] = lucro_politica
    cenarios["lucro_baseline"] = lucro_baseline
    return metricas, cenarios


# --- 6) Tracking (MLflow/DagsHub opcional, fail-closed) -----------------------

_PROJECT_SLUG = "Otimizacao_de_Receita"
# Prefixo de variáveis de ambiente DESTE projeto. Namespacear é o que impede
# herdar credenciais globais de OUTROS projetos DagsHub na mesma máquina.
_ENV_PREFIX = "OTIMIZACAO_"
# Credenciais genéricas (compartilháveis entre projetos) que NUNCA devem ser
# reaproveitadas silenciosamente aqui.
_ENV_GENERICAS_CRED: tuple[str, ...] = (
    "MLFLOW_TRACKING_TOKEN",
    "MLFLOW_TRACKING_PASSWORD",
    "DAGSHUB_TOKEN",
)


def _env(*nomes: str) -> str | None:
    """Primeiro valor não-vazio entre os nomes de variável de ambiente dados."""
    for nome in nomes:
        valor = os.getenv(nome)
        if valor:
            return valor
    return None


def _belongs_to_this_project(uri: str | None) -> bool:
    """Só aceita tracking remoto que comprovadamente pertence a este projeto."""
    return _PROJECT_SLUG.lower() in (uri or "").lower()


def init_tracking() -> None:
    """Configura MLflow: local por padrão; remoto só com config deste projeto.

    Fail-closed contra contaminação entre projetos (a máquina pode hospedar
    credenciais de vários projetos DagsHub):

    1. **Destino** — o repo/URI remoto precisa conter o slug deste projeto; caso
       contrário, usamos ``mlruns/`` local.
    2. **Credenciais** — só são aceitas as namespaced (``OTIMIZACAO_...``).
       Variáveis genéricas (``DAGSHUB_TOKEN``, ``MLFLOW_TRACKING_PASSWORD``) são
       deliberadamente ignoradas por poderem pertencer a outro projeto, e
       removidas do ambiente de autenticação do MLflow.
    """
    repo = _env(f"{_ENV_PREFIX}DAGSHUB_REPO", "DAGSHUB_REPO_NAME", "DAGSHUB_REPO")
    owner = _env(f"{_ENV_PREFIX}DAGSHUB_OWNER", "DAGSHUB_REPO_OWNER", "DAGSHUB_OWNER")
    tracking_uri_env = _env(f"{_ENV_PREFIX}MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_URI")

    # Credenciais: apenas as específicas deste projeto.
    usuario = _env(f"{_ENV_PREFIX}MLFLOW_TRACKING_USERNAME")
    senha = _env(f"{_ENV_PREFIX}MLFLOW_TRACKING_PASSWORD")
    token = _env(f"{_ENV_PREFIX}MLFLOW_TRACKING_TOKEN", f"{_ENV_PREFIX}DAGSHUB_TOKEN")
    tem_credenciais = bool(token) or (bool(usuario) and bool(senha))
    genericas_presentes = [nome for nome in _ENV_GENERICAS_CRED if os.getenv(nome)]

    def _limpar_credenciais_genericas() -> None:
        """Zera credenciais globais p/ o MLflow não autenticar com outro projeto."""
        for nome in (*_ENV_GENERICAS_CRED, "MLFLOW_TRACKING_USERNAME"):
            os.environ.pop(nome, None)

    def _local() -> bool:
        try:
            mlflow.set_tracking_uri((Path.cwd() / "mlruns").resolve().as_uri())
            return True
        except Exception as erro:  # pragma: no cover - só em ambientes exóticos
            warnings.warn(
                f"Falha ao definir tracking local do MLflow: {erro}", stacklevel=2
            )
            return False

    remoto: str | None = None
    if tracking_uri_env and _belongs_to_this_project(tracking_uri_env):
        remoto = tracking_uri_env
    elif owner and repo and _belongs_to_this_project(repo):
        remoto = f"https://dagshub.com/{owner}/{repo}.mlflow"
    elif tracking_uri_env:
        warnings.warn(
            f"MLFLOW_TRACKING_URI ('{tracking_uri_env}') aponta para outro projeto; "
            "ignorado para não contaminar o tracking alheio. Usando mlruns/ local.",
            stacklevel=2,
        )

    usar_remoto = False
    if remoto and tem_credenciais:
        # Espelha APENAS as credenciais deste projeto (nunca deixa sobras globais).
        if usuario:
            os.environ["MLFLOW_TRACKING_USERNAME"] = usuario
        else:
            os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
        if senha:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = senha
        else:
            os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
        if token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = token
        else:
            os.environ.pop("MLFLOW_TRACKING_TOKEN", None)
        try:
            mlflow.set_tracking_uri(remoto)
            usar_remoto = True
        except Exception as erro:  # pragma: no cover - rede/credenciais
            warnings.warn(
                f"Falha ao definir tracking remoto ({erro}); usando mlruns/ local.",
                stacklevel=2,
            )

    if not usar_remoto:
        if remoto and not tem_credenciais:
            warnings.warn(
                "Tracking remoto deste projeto detectado, mas sem credenciais "
                f"namespaced ('{_ENV_PREFIX}MLFLOW_TRACKING_TOKEN' ou "
                f"'{_ENV_PREFIX}DAGSHUB_TOKEN'). Usando mlruns/ local.",
                stacklevel=2,
            )
        elif genericas_presentes:
            warnings.warn(
                "Credenciais DagsHub genéricas detectadas "
                f"({', '.join(genericas_presentes)}), mas sem versões namespaced "
                f"('{_ENV_PREFIX}...'); ignoradas para não contaminar entre projetos. "
                "Usando mlruns/ local.",
                stacklevel=2,
            )
        _local()
        _limpar_credenciais_genericas()

    try:
        mlflow.set_experiment(
            os.getenv("MLFLOW_EXPERIMENT_NAME", "otimizacao_preco_demanda")
        )
        print(f"[MLflow] Tracking URI: {mlflow.get_tracking_uri()}")
    except Exception as erro:  # pragma: no cover - servidor indisponível
        warnings.warn(
            f"Falha ao configurar experimento no MLflow: {erro}", stacklevel=2
        )


def _registrar_candidato(
    candidato: dict[str, object],
    campeao: str,
    resultado: dict[str, object],
    caminhos: list[Path],
) -> None:
    """Um run de MLflow por candidato; o campeão recebe a tag ``model_best``.

    Loga os hiperparâmetros e as métricas de validação (factuais + negócio). No
    run campeão, anexa também as métricas do teste (Q4) e os artefatos — de modo
    que o ``model_best`` fique totalmente rastreável no DagsHub/MLflow.
    """
    nome = str(candidato["modelo"])
    metricas = candidato["validacao"]  # type: ignore[assignment]
    params = candidato["params"]  # type: ignore[assignment]

    with mlflow.start_run(run_name=nome):
        mlflow.log_params({"modelo": nome, **params})
        mlflow.set_tag("fase", "fase3-modelagem")
        mlflow.set_tag("criterio_selecao", "lucro_politica_validacao")
        e_campeao = nome == campeao
        mlflow.set_tag("model_best", str(e_campeao).lower())
        mlflow.log_metrics(
            {
                "validacao_rmse": metricas["rmse"],
                "validacao_mae": metricas["mae"],
                "validacao_r2": metricas["r2"],
                "validacao_lucro_politica": metricas["lucro_politica"],
                "validacao_uplift_pct": metricas["uplift_pct"],
                "validacao_desconto_medio": metricas["desconto_medio"],
            }
        )
        if e_campeao:
            mlflow.log_metrics(
                {f"teste_{chave}": valor for chave, valor in resultado["teste"].items()}  # type: ignore[union-attr]
            )
            for caminho in caminhos:
                with contextlib.suppress(Exception):
                    mlflow.log_artifact(str(caminho))


def registrar_experimentos(resultado: dict[str, object], caminhos: list[Path]) -> None:
    """Publica cada candidato no MLflow e marca o campeão como ``model_best``.

    Best-effort: o tracking nunca derruba o pipeline — os artefatos já foram
    publicados em disco. Com credenciais do DagsHub no ambiente, os runs
    chegam ao servidor remoto; sem elas, ficam no ``mlruns/`` local.
    """
    for candidato in resultado["candidatos"]:  # type: ignore[union-attr]
        with contextlib.suppress(Exception):
            _registrar_candidato(
                candidato, str(resultado["modelo"]), resultado, caminhos
            )
    print(f"[MLflow] Runs registrados: {len(resultado['candidatos'])} candidatos")  # type: ignore[arg-type]


# --- 7) Orquestração ----------------------------------------------------------


def treinar_e_avaliar(df: pd.DataFrame) -> dict[str, object]:
    """Executa o ciclo completo e devolve o pacote de resultados do pipeline.

    Não escreve artefatos: é a função pura/testável do pipeline. A persistência
    fica em `main()`.
    """
    validar_contrato_features(df)
    splits = separar_splits(df)
    treino, validacao, teste = splits["treino"], splits["validacao"], splits["teste"]

    X_treino, y_treino = (
        treino[list(FEATURES_NUMERICAS + FEATURES_CATEGORICAS)],
        treino[TARGET],
    )
    X_val, y_val = (
        validacao[list(FEATURES_NUMERICAS + FEATURES_CATEGORICAS)],
        validacao[TARGET],
    )
    X_teste, y_teste = (
        teste[list(FEATURES_NUMERICAS + FEATURES_CATEGORICAS)],
        teste[TARGET],
    )

    candidatos = construir_candidatos(_largura_onehot(X_treino))
    resultados: list[dict[str, object]] = []
    melhor: ModelSpec | None = None
    melhor_lucro = -float("inf")

    for spec in candidatos:
        spec.pipeline.fit(X_treino, y_treino)
        metricas_val = {
            **metricas_demanda(
                y_val.to_numpy(dtype=float), spec.pipeline.predict(X_val)
            ),
            **metricas_negocio(spec.pipeline, X_val),
        }
        resultados.append(
            {"modelo": spec.nome, "params": spec.params, "validacao": metricas_val}
        )
        print(
            f"[validação] {spec.nome:<22} lucro={metricas_val['lucro_politica']:8.2f} "
            f"uplift={metricas_val['uplift_pct']:6.2f}%  rmse={metricas_val['rmse']:.3f}"
        )
        if metricas_val["lucro_politica"] > melhor_lucro:
            melhor_lucro = metricas_val["lucro_politica"]
            melhor = spec

    assert melhor is not None, "Nenhum modelo foi avaliado."

    # Re-ajuste do campeão em treino + validação (mais passado, mesmo teste).
    passado = pd.concat([treino, validacao], ignore_index=True)
    X_passado, y_passado = (
        passado[list(FEATURES_NUMERICAS + FEATURES_CATEGORICAS)],
        passado[TARGET],
    )
    melhor.pipeline.fit(X_passado, y_passado)

    metricas_teste, cenarios = _enriquecer(melhor.pipeline, teste)
    baseline = baselines_ingenuos(passado, X_teste, y_teste.to_numpy(dtype=float))
    importancias = importancias_por_feature(melhor.pipeline, X_passado, y_passado)
    curva = _curva_negocio(melhor.pipeline, teste)

    return {
        "modelo": melhor.nome,
        "params": melhor.params,
        "estimador": melhor.pipeline,
        "candidatos": resultados,
        "validacao": next(
            r["validacao"] for r in resultados if r["modelo"] == melhor.nome
        ),
        "teste": metricas_teste,
        "baselines": baseline,
        "importancias": importancias,
        "curva_negocio": curva,
        "cenarios": cenarios,
        "splits": {nome: int(len(parte)) for nome, parte in splits.items()},
        "periodos": {
            "treino": _periodo(treino),
            "validacao": _periodo(validacao),
            "teste": _periodo(teste),
        },
        "hash_dados": _hash_dados(passado),
    }


def _periodo(df: pd.DataFrame) -> list[str]:
    return [str(df["data"].min().date()), str(df["data"].max().date())]


def _proveniencia(resultado: dict[str, object]) -> dict[str, object]:
    return {
        "seed": SEED,
        "gerado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": os.getenv("GITHUB_SHA", "local"),
        "hash_dados": resultado["hash_dados"],
        "linhas_por_split": resultado["splits"],
        "periodos": resultado["periodos"],
    }


def _nota_honestidade() -> str:
    return (
        "Métricas de demanda no teste são factuais (demanda observada vs. prevista). "
        "O lucro da política é uma estimativa contrafactual (a demanda sob descontos "
        "alternativos é prevista pelo modelo), não um resultado realizado; use o "
        "baseline ingênuo como calibração do ganho esperado."
    )


# --- 7) Bundle de serving do app (CRISP-DM — Deployment) ----------------------

# Artefatos pequenos, versionados no Git, que o app (Streamlit Cloud) consome
# sem DVC nem a base crua. O `test_predictions.parquet` fica de fora: é
# diagnóstico pesado e permanece sob governança do DVC em `models/`.
BUNDLE_APP: tuple[str, ...] = (
    "best_model.pkl",
    "model_card.json",
    "metrics_snapshot.json",
    "business_curve.csv",
)


def catalogo_app(df: pd.DataFrame) -> dict[str, object]:
    """Domínios das categóricas + cenários-tipo para o app montar linhas válidas.

    Os templates vêm apenas de treino+validação (o "passado"), nunca do teste,
    para não vazar o futuro para dentro do simulador. Cada template é a última
    observação de um par (loja, produto), isto é, um vetor de features **real e
    coerente** — o app o usa como base e só sobrescreve as alavancas de decisão.
    """
    passado = df[df["split"].isin(("treino", "validacao"))]
    features = list(FEATURES_NUMERICAS + FEATURES_CATEGORICAS)
    dominios = {
        coluna: sorted(passado[coluna].dropna().unique().tolist())
        for coluna in FEATURES_CATEGORICAS
    }
    ultima = (
        passado.sort_values("data").groupby(["loja", "produto"], as_index=False).last()
    )
    # `to_json` normaliza os tipos numpy/pandas e converte NaN em `null`.
    templates = sorted(
        json.loads(ultima[features].to_json(orient="records")),
        key=lambda linha: (linha["loja"], linha["produto"]),
    )
    return {
        "alvo": TARGET,
        "features_numericas": list(FEATURES_NUMERICAS),
        "features_categoricas": list(FEATURES_CATEGORICAS),
        "grade_desconto": [float(d) for d in GRADE_DESCONTO],
        "dominios": dominios,
        "templates": templates,
        "colunas_vetadas_por_vazamento": list(colunas_com_vazamento()),
    }


def publicar_bundle_app(df: pd.DataFrame) -> Path:
    """Publica o bundle de serving (catálogo + cópias pequenas) em `app/artifacts/`."""
    APP_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    catalogo = APP_ARTIFACTS_DIR / "app_catalog.json"
    catalogo.write_text(
        json.dumps(catalogo_app(df), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for nome in BUNDLE_APP:
        (APP_ARTIFACTS_DIR / nome).write_bytes((MODELS_DIR / nome).read_bytes())
    return APP_ARTIFACTS_DIR


def main() -> None:
    """Executa o pipeline de modelagem e persiste o contrato de artefatos."""
    init_tracking()
    df = carregar_base_preparada()
    resultado = treinar_e_avaliar(df)
    estimador: Pipeline = resultado["estimador"]  # type: ignore[assignment]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # (1) Modelo campeão: pipeline completo (preproc + estimador), pronto p/ serving.
    caminho_modelo = MODELS_DIR / "best_model.pkl"
    joblib.dump(estimador, caminho_modelo)

    # (2) Snapshot de métricas (contrato consumido por testes, site e app).
    metricas = {
        "alvo": TARGET,
        "modelo": resultado["modelo"],
        "validacao": resultado["validacao"],
        "teste": resultado["teste"],
        "baselines": resultado["baselines"],
        "candidatos": resultado["candidatos"],
        "proveniencia": _proveniencia(resultado),
        "honestidade": _nota_honestidade(),
    }
    caminho_metricas = MODELS_DIR / "metrics_snapshot.json"
    caminho_metricas.write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # (3) Cartão de modelo: esquema de features e como reusar (Fase 4 / app).
    cartao = {
        "alvo": TARGET,
        "modelo": resultado["modelo"],
        "objetivo": "Escolher, por SKU/dia, o desconto que maximiza o lucro esperado.",
        "features_numericas": list(FEATURES_NUMERICAS),
        "features_categoricas": list(FEATURES_CATEGORICAS),
        "colunas_vetadas_por_vazamento": list(colunas_com_vazamento()),
        "grade_desconto": [float(d) for d in GRADE_DESCONTO],
        "restricao_preco_minimo": "Otimizador respeita `preco_minimo` por linha.",
        "importancia_features": resultado["importancias"],
        "metricas_teste": resultado["teste"],
        "honestidade": _nota_honestidade(),
        "proveniencia": _proveniencia(resultado),
    }
    caminho_cartao = MODELS_DIR / "model_card.json"
    caminho_cartao.write_text(
        json.dumps(cartao, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # (4) Curva de negócio (desconto -> demanda/receita/lucro) para o site/app.
    caminho_curva = MODELS_DIR / "business_curve.csv"
    resultado["curva_negocio"].to_csv(caminho_curva, index=False)  # type: ignore[union-attr]

    # (5) Cenários por linha no teste (diagnóstico honesto do notebook 05).
    caminho_cenarios = MODELS_DIR / "test_predictions.parquet"
    resultado["cenarios"].to_parquet(caminho_cenarios, index=False)  # type: ignore[union-attr]

    # (6) Bundle de serving do app: catálogo + cópias pequenas versionadas no Git.
    destino_app = publicar_bundle_app(df)

    # (7) Tracking: um run por candidato no MLflow/DagsHub, campeão = model_best.
    registrar_experimentos(
        resultado,
        [
            caminho_modelo,
            caminho_metricas,
            caminho_cartao,
            caminho_curva,
            caminho_cenarios,
        ],
    )

    print(
        f"\nCampeão: {resultado['modelo']}\n"
        f"Teste (demanda): rmse={resultado['teste']['rmse']:.3f} "
        f"mae={resultado['teste']['mae']:.3f} r2={resultado['teste']['r2']:.3f}\n"
        f"Teste (negócio): uplift={resultado['teste']['uplift_pct']:.2f}% "
        f"desconto médio={resultado['teste']['desconto_medio']:.3f}\n"
        f"Artefatos em: {MODELS_DIR}\n"
        f"Bundle do app em: {destino_app}"
    )


if __name__ == "__main__":
    main()
