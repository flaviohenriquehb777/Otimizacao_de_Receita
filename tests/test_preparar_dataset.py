"""Testes da etapa CRISP-DM de Data Preparation (base perfeita + anti-leakage).

Cobrem três garantias críticas do projeto:
1. A validação é fail-closed: aprova base íntegra e reprova preço inconsistente.
2. O split temporal é estrito: nenhum registro futuro entra no treino.
3. O guardrail de vazamento remove colunas agregadas do próprio dia-alvo.
"""

import pandas as pd
import pytest

from src.config.paths import DATA_DIR
from src.data import preparar_dataset as prep

BRUTO_CANONICO = DATA_DIR / "dataset_cafeterias_rj.parquet"


def _df_exemplo(datas: list[str], desconto: float = 0.10) -> pd.DataFrame:
    """Base mínima e consistente (1 loja x 1 produto x datas informadas)."""
    preco_original = 100.0
    preco_final = round(preco_original * (1.0 - desconto), 2)
    registros = [
        {
            "data": pd.Timestamp(d),
            "loja": "Unidade A",
            "produto": "Torta Standard 10 fatias",
            "categoria": "Torta",
            "custo_producao": 35.0,
            "preco_original": preco_original,
            "desconto_pct": desconto,
            "preco_final": preco_final,
            "preco_competidor": 95.0,
            "quantidade_vendida_dia": 10,
            "quantidade_vendida_mes": 300,  # agregado do próprio mês (vazamento)
            "estoque_disponivel": 14,  # derivado da demanda do dia (vazamento)
            "ruptura_flag": 0,  # desfecho do dia (vazamento)
            "custo_marketing_unitario": 5.0,  # gasto/quantidade (vazamento)
            "promocao_flag": 0,
        }
        for d in datas
    ]
    return pd.DataFrame(registros)


def test_validacao_aprova_base_consistente():
    assert prep.validar_esquema(_df_exemplo(["2024-01-01", "2024-01-02"])) == []


def test_validacao_reprova_preco_final_inconsistente():
    df = _df_exemplo(["2024-01-01"])
    df.loc[0, "preco_final"] = 42.0  # divergente de 100.0 * (1 - 0.10)
    problemas = prep.validar_esquema(df)
    assert any("preco_final" in p for p in problemas)
    with pytest.raises(ValueError):
        prep.assert_validacao(df)


def test_vazamento_excluido_das_features():
    df = prep.marcar_split_temporal(
        prep.engenharia_features(prep.limpar_dados(_df_exemplo(["2024-01-01"])))
    )
    features = prep.selecionar_features(df)
    # O guardrail precisa cobrir agregados temporais E desfechos do dia-alvo.
    for coluna in (
        "quantidade_vendida_mes",
        "estoque_disponivel",
        "ruptura_flag",
        "custo_marketing_unitario",
    ):
        assert coluna in prep.colunas_com_vazamento(), f"{coluna} ausente do guardrail."
        assert coluna not in features, f"{coluna} vazou para as features."
    # Alvos nunca entram como preditoras.
    for alvo in (
        "receita_dia",
        "lucro_dia",
        "margem_unitaria",
        "quantidade_vendida_dia",
    ):
        assert alvo not in features


def test_split_temporal_estrito():
    df = _df_exemplo(["2025-06-30", "2025-07-01", "2025-09-30", "2025-10-01"])
    df = prep.marcar_split_temporal(df)
    split = dict(zip(df["data"].dt.strftime("%Y-%m-%d"), df["split"], strict=True))
    assert split["2025-06-30"] == "treino"
    assert split["2025-07-01"] == "validacao"
    assert split["2025-09-30"] == "validacao"
    assert split["2025-10-01"] == "teste"


def test_split_sem_futuro_no_treino():
    datas = (
        pd.date_range("2024-01-01", "2025-12-31", freq="D")
        .strftime("%Y-%m-%d")
        .tolist()
    )
    df = prep.marcar_split_temporal(_df_exemplo(datas))
    treino = df[df["split"] == "treino"]
    assert treino["data"].max() <= pd.Timestamp(prep.DATA_CORTE_TREINO)
    assert treino["data"].max() < df[df["split"] == "teste"]["data"].min()


def test_features_derivadas_sao_consistentes():
    df = prep.engenharia_features(
        prep.limpar_dados(_df_exemplo(["2024-01-01"], desconto=0.20))
    )
    linha = df.iloc[0]
    assert linha["receita_dia"] == pytest.approx(
        linha["preco_final"] * linha["quantidade_vendida_dia"]
    )
    assert linha["lucro_dia"] == pytest.approx(
        (linha["preco_final"] - linha["custo_producao"])
        * linha["quantidade_vendida_dia"]
    )
    assert 0.0 <= linha["margem_unitaria"] <= 1.0


@pytest.mark.skipif(
    not BRUTO_CANONICO.exists(), reason="Base bruta canônica ausente (rode dvc repro)."
)
def test_integracao_base_canonica_perfeita():
    df = prep.carregar_dados_brutos()
    assert (
        prep.validar_esquema(df) == []
    ), "A base canônica bruta deve passar na validação."
    preparado = prep.preparar()
    assert set(preparado["split"].unique()) == {"treino", "validacao", "teste"}
    assert preparado["data"].is_monotonic_increasing

    # Os únicos nulos permitidos são as datas de promoção ausente (semânticos).
    nulos = preparado.isna().sum()
    colunas_com_nulo = set(nulos[nulos > 0].index)
    assert colunas_com_nulo <= set(
        prep.colunas_nulas_intencionais()
    ), f"Nulos inesperados em: {colunas_com_nulo - set(prep.colunas_nulas_intencionais())}"
    assert prep.resumo_qualidade(preparado)["nulos_totais"] == 0


@pytest.mark.skipif(
    not BRUTO_CANONICO.exists(), reason="Base bruta canônica ausente (rode dvc repro)."
)
def test_auditoria_de_vazamento_na_base_canonica():
    """Regressão da auditoria: as colunas vetadas realmente carregam sinal do alvo."""
    preparado = prep.preparar()
    alvo = preparado["quantidade_vendida_dia"]

    # Agregado temporal e desfecho derivado da demanda têm sinal forte com o alvo.
    for coluna, limite in (
        ("quantidade_vendida_mes", 0.5),
        ("estoque_disponivel", 0.5),
    ):
        corr = abs(float(preparado[coluna].corr(alvo)))
        assert corr > limite, f"{coluna} deveria vazar o alvo (|corr|={corr:.3f})."

    # E todo o guardrail precisa ficar fora das features.
    assert set(prep.colunas_com_vazamento()).isdisjoint(
        prep.selecionar_features(preparado)
    )
