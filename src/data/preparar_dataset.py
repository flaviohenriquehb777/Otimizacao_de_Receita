"""Preparação canônica do dataset (CRISP-DM — Data Preparation).

Motivação
---------
O dataset bruto (`dataset_cafeterias_rj.parquet`) é gerado pelo simulador
calibrado e já vem íntegro por construção. Ainda assim, a etapa de *Data
Preparation* existe para:

1. **Blindar a verdade dos dados**: validar esquema, regras de negócio e
   consistência aritmética (ex.: `preco_final == preco_original * (1 - desconto)`).
2. **Padronizar tipos e derivar features** de forma determinística e testável,
   em vez de espalhar lógica por células de notebook.
3. **Prevenir vazamento de dados (data leakage)**: expor explicitamente as
   colunas que carregam sinal do alvo (agregados e desfechos) e **não** devem
   entrar como preditoras, além de materializar o **split temporal estrito**
   dentro do próprio dataset.

Guardrail de vazamento
----------------------
Uma auditoria com detector de 1 feature (R² de um modelo linear de uma variável
sobre o alvo) confirmou **quatro** colunas que carregam sinal do alvo
`quantidade_vendida_dia` e, por isso, são vetadas como preditoras:

* `quantidade_vendida_mes` — agregado do mês que inclui o próprio dia (vaza alvo e futuro).
* `estoque_disponivel` — derivado da demanda realizada no dia (R² = 0,95).
* `ruptura_flag` — consequência da demanda vs. estoque do dia (97,8% reconstruível).
* `custo_marketing_unitario` — `gasto_marketing / quantidade` (identidade exata).

Elas permanecem na base para relatórios e auditoria, porém são listadas em
`colunas_com_vazamento()` e excluídas de `selecionar_features()` por padrão.

Reprodutibilidade
-----------------
Determinístico com `SEED = 42`; o split temporal usa datas fixas, nunca
amostragem aleatória.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import DADOS_CAFETERIAS_PREPARADO, DATA_DIR

SEED = 42
SAIDA_PARQUET = DADOS_CAFETERIAS_PREPARADO

# --- Split temporal estrito (anti-leakage) -----------------------------------
# Treino enxerga o passado; teste cobre o pico de fim de ano (Q4), o cenário
# mais adverso para generalização. Datas fixas: nada de shuffle aleatório.
DATA_CORTE_TREINO = "2025-06-30"  # <= treino
DATA_CORTE_VALIDACAO = "2025-09-30"  # (treino, validação]; teste = > validação

# Colunas que carregam sinal do alvo e, portanto, **nunca** podem ser preditoras.
#
# A auditoria da base (detector de 1 feature, isto é, R² de um modelo linear com
# uma única variável sobre o alvo) revelou três fontes distintas de vazamento:
#
# 1. Agregado temporal que inclui o próprio dia-alvo:
#    - `quantidade_vendida_mes` — soma do mês inteiro (passado **e** futuro) do
#      mesmo (mês, loja, produto). R² = 0,73. Vaza alvo **e** futuro.
# 2. Variáveis derivadas da demanda realizada no próprio dia:
#    - `estoque_disponivel`     — gerado como `demanda × uniform(0,85; 1,65)`. R² = 0,95.
#    - `ruptura_flag`           — `estoque < quantidade` (consequência da demanda); 97,8% reconstruível.
#    - `custo_marketing_unitario` — `gasto_marketing / quantidade` (identidade exata). R² = 0,30.
#
# Em dados transacionais reais estas colunas existem como **desfecho** do dia, não
# como insumo da decisão de preço — que ocorre *antes* do dia. Mantemo-las na base
# para relatórios/auditoria, mas fora das features.
_COLUNAS_VAZAMENTO: tuple[str, ...] = (
    "quantidade_vendida_mes",
    "estoque_disponivel",
    "custo_marketing_unitario",
    "ruptura_flag",
)

# Colunas obrigatórias no dataset bruto.
_COLUNAS_OBRIGATORIAS: tuple[str, ...] = (
    "data",
    "loja",
    "produto",
    "categoria",
    "custo_producao",
    "preco_original",
    "desconto_pct",
    "preco_final",
    "quantidade_vendida_dia",
    "promocao_flag",
)

# Tolerância para revalidar a consistência de preço final (arredondamento).
_TOL_PRECO = 0.02

# Colunas de data de promoção ficam nulas (NaT) quando não há promoção ativa.
# Isso é semântico, não falta de dado: imputar aqui criaria informação falsa.
_COLUNAS_NULAS_INTENCIONAIS: tuple[str, ...] = ("promocao_inicio", "promocao_fim")


def carregar_dados_brutos(caminho: Path | None = None) -> pd.DataFrame:
    """Carrega o dataset bruto canônico (Parquet) gerado pelo simulador."""
    caminho = caminho or (DATA_DIR / "dataset_cafeterias_rj.parquet")
    if not caminho.exists():
        raise FileNotFoundError(
            f"Dataset bruto não encontrado em {caminho}. "
            "Rode 'python -m src.data.gerar_dataset' (ou 'dvc repro gerar_dados')."
        )
    return pd.read_parquet(caminho)


def validar_esquema(df: pd.DataFrame) -> list[str]:
    """Valida o esquema e as regras de negócio; retorna a lista de problemas.

    Uma lista vazia significa que a base está íntegra. Não lança exceção para
    permitir que os notebooks apresentem o relatório de qualidade; o pipeline
    pode optar por falhar (fail-closed) via `assert_validacao()`.
    """
    problemas: list[str] = []

    faltantes = [c for c in _COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltantes:
        problemas.append(f"Colunas obrigatórias ausentes: {faltantes}")
        return problemas  # sem esquema mínimo, demais checagens não fazem sentido

    if df["data"].isna().any():
        problemas.append("Coluna 'data' contém valores nulos/não parseáveis.")
    if df.duplicated(subset=["data", "loja", "produto"]).any():
        problemas.append("Existem duplicatas na chave (data, loja, produto).")
    if (df["desconto_pct"] < 0).any() or (df["desconto_pct"] > 1).any():
        problemas.append("'desconto_pct' fora do intervalo [0, 1].")
    if (df["preco_original"] <= 0).any():
        problemas.append("'preco_original' deve ser estritamente positivo.")
    if (df["custo_producao"] < 0).any():
        problemas.append("'custo_producao' não pode ser negativo.")
    if (df["quantidade_vendida_dia"] < 0).any():
        problemas.append("'quantidade_vendida_dia' não pode ser negativa.")

    # Consistência aritmética do preço final.
    esperado = df["preco_original"] * (1.0 - df["desconto_pct"])
    divergencia = (df["preco_final"] - esperado).abs() > _TOL_PRECO
    if divergencia.any():
        problemas.append(
            f"'preco_final' inconsistente com preco_original*(1-desconto) em "
            f"{int(divergencia.sum())} linha(s)."
        )

    return problemas


def assert_validacao(df: pd.DataFrame) -> None:
    """Falha (fail-closed) se a base não passar na validação de esquema."""
    problemas = validar_esquema(df)
    if problemas:
        raise ValueError("Falha na validação do dataset:\n- " + "\n- ".join(problemas))


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza a base: tipos, regras de negócio e consistência aritmética."""
    df = df.copy()

    # Tipos e ordenação canônicos.
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data"])
    df = df.drop_duplicates(subset=["data", "loja", "produto"])

    colunas_numericas = [
        "custo_producao",
        "preco_original",
        "desconto_pct",
        "preco_final",
        "quantidade_vendida_dia",
        "preco_competidor",
        "indice_preco_competidor",
        "estoque_disponivel",
        "gasto_marketing",
    ]
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Regras de negócio: desconto em [0, 0.6]; preços e custos não negativos.
    df["desconto_pct"] = df["desconto_pct"].clip(lower=0.0, upper=0.6)
    for col in ("preco_original", "preco_final", "custo_producao"):
        df[col] = df[col].clip(lower=0.0)

    # Reimputa preco_final para garantir consistência aritmética exata.
    df["preco_final"] = (df["preco_original"] * (1.0 - df["desconto_pct"])).round(2)

    # Imputação defensiva de numéricos críticos (não deve haver NaN no bruto).
    for col in ("custo_producao", "preco_original", "preco_final"):
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df["quantidade_vendida_dia"] = (
        df["quantidade_vendida_dia"].fillna(0).astype("int64")
    )

    # Categóricos: nulos viram rótulo explícito (não quebra one-hot/modelo).
    for col in ("evento_especial", "promocao_tipo", "campanha_id"):
        if col in df.columns:
            df[col] = df[col].fillna("Nenhum").astype(str)

    return df.sort_values(["data", "loja", "produto"]).reset_index(drop=True)


def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva features de negócio determinísticas e sem vazamento.

    Nenhuma feature usa informação futura nem o agregado do próprio dia-alvo.
    """
    df = df.copy()

    # Métricas transacionais do dia (alvos e apoio à decisão).
    df["receita_dia"] = (df["preco_final"] * df["quantidade_vendida_dia"]).round(2)
    df["lucro_dia"] = (
        (df["preco_final"] - df["custo_producao"]) * df["quantidade_vendida_dia"]
    ).round(2)
    df["margem_unitaria"] = np.where(
        df["preco_final"] > 0,
        (df["preco_final"] - df["custo_producao"]) / df["preco_final"],
        0.0,
    ).round(4)

    # Calendário derivado (determinístico a partir da data).
    df["ano"] = df["data"].dt.year
    df["trimestre"] = df["data"].dt.quarter
    df["dia_semana_idx"] = df["data"].dt.dayofweek
    df["fim_de_semana"] = (df["dia_semana_idx"] >= 5).astype("int64")

    # Razão preço vs. competidor (proxy de competitividade do dia).
    if "indice_preco_competidor" not in df.columns and "preco_competidor" in df.columns:
        df["indice_preco_competidor"] = (
            df["preco_final"] / df["preco_competidor"]
        ).round(4)

    return df


def colunas_com_vazamento() -> tuple[str, ...]:
    """Colunas que carregam sinal do alvo (agregado/desfecho) — nunca features."""
    return _COLUNAS_VAZAMENTO


def colunas_nulas_intencionais() -> tuple[str, ...]:
    """Colunas que podem ficar nulas legitimamente (sem promoção ativa)."""
    return _COLUNAS_NULAS_INTENCIONAIS


def selecionar_features(df: pd.DataFrame, incluir_vazamento: bool = False) -> list[str]:
    """Lista as colunas candidatas a preditoras, excluindo alvos e vazamento."""
    alvos = {
        "quantidade_vendida_dia",
        "receita_dia",
        "lucro_dia",
        "margem_unitaria",
    }
    proibidas = set(colunas_com_vazamento()) if not incluir_vazamento else set()

    features = [
        col
        for col in df.columns
        if col not in alvos and col not in proibidas and col != "data"
    ]
    return features


def marcar_split_temporal(
    df: pd.DataFrame,
    corte_treino: str = DATA_CORTE_TREINO,
    corte_validacao: str = DATA_CORTE_VALIDACAO,
) -> pd.DataFrame:
    """Materializa o split temporal estrito na coluna 'split'.

    Garante a fronteira passado/futuro: nenhuma linha do futuro entra no treino.
    """
    df = df.copy()
    corte_treino_ts = pd.Timestamp(corte_treino)
    corte_validacao_ts = pd.Timestamp(corte_validacao)

    condicoes = [
        df["data"] <= corte_treino_ts,
        df["data"] <= corte_validacao_ts,
    ]
    df["split"] = np.select(condicoes, ["treino", "validacao"], default="teste").astype(
        str
    )
    return df


def resumo_qualidade(df: pd.DataFrame) -> dict[str, object]:
    """Resume a qualidade e a distribuição temporal da base preparada."""
    receita_total = (
        float(df["receita_dia"].sum()) if "receita_dia" in df else float("nan")
    )
    nulos = df.isna().sum()
    nulos_intencionais = int(nulos.reindex(_COLUNAS_NULAS_INTENCIONAIS).fillna(0).sum())
    return {
        "linhas": int(len(df)),
        "colunas": int(df.shape[1]),
        "periodo_inicio": str(df["data"].min().date()),
        "periodo_fim": str(df["data"].max().date()),
        "lojas": sorted(df["loja"].unique().tolist()),
        "produtos": int(df["produto"].nunique()),
        "receita_total": round(receita_total, 2),
        "nulos_totais": int(nulos.sum()) - nulos_intencionais,
        "nulos_intencionais": nulos_intencionais,
        "split_contagem": df["split"].value_counts().to_dict(),
    }


def preparar(caminho_bruto: Path | None = None) -> pd.DataFrame:
    """Orquestra a preparação: valida -> limpa -> features -> split temporal."""
    df = carregar_dados_brutos(caminho_bruto)
    assert_validacao(df)  # fail-closed: a base precisa estar perfeita.
    df = limpar_dados(df)
    df = engenharia_features(df)
    df = marcar_split_temporal(df)
    return df


def salvar_parquet(df: pd.DataFrame, destino: Path = SAIDA_PARQUET) -> Path:
    """Persiste a base preparada em Parquet (governança via DVC)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)
    return destino


def main() -> None:
    """Prepara, resume e persiste a base canônica do projeto."""
    df = preparar()
    destino = salvar_parquet(df)
    resumo = resumo_qualidade(df)
    print(f"Dataset preparado: {destino}")
    print(f"Linhas: {resumo['linhas']:,} | Colunas: {resumo['colunas']}")
    print(f"Periodo: {resumo['periodo_inicio']} a {resumo['periodo_fim']}")
    print(f"Receita total: R$ {resumo['receita_total']:,.2f}")
    print(f"Split: {resumo['split_contagem']}")
    print(
        f"Nulos inesperados: {resumo['nulos_totais']} | "
        f"nulos intencionais (sem promoção): {resumo['nulos_intencionais']}"
    )
    print(f"Colunas com vazamento (excluídas das features): {colunas_com_vazamento()}")


if __name__ == "__main__":
    main()
