"""Contrato de artefatos e guardrail anti-vazamento do pipeline de modelagem.

O treino é executado uma única vez (fixture de módulo) e os testes verificam o
*contrato* publicado — incluindo a garantia de que nenhum artefato reintroduz
as colunas vetadas por vazamento.
"""

import json

import pandas as pd
import pytest

from src.config.paths import DOCS_DIR, MODELS_DIR
from src.data.preparar_dataset import colunas_com_vazamento
from src.modeling.train_pipeline import (
    BUNDLE_DOCS,
    FEATURES_CATEGORICAS,
    FEATURES_NUMERICAS,
    GRADE_DESCONTO,
    TARGET,
)


def _carregar_json(nome: str) -> dict:
    caminho = MODELS_DIR / nome
    assert caminho.exists(), f"Artefato ausente: {nome}"
    return json.loads(caminho.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artefatos(treino_executado: None) -> dict:
    """Lê o snapshot + o cartão publicados pelo pipeline (treino roda 1x/sessão)."""
    return {
        "snapshot": _carregar_json("metrics_snapshot.json"),
        "cartao": _carregar_json("model_card.json"),
    }


def test_pipeline_publica_o_contrato_de_artefatos(artefatos: dict) -> None:
    assert (MODELS_DIR / "best_model.pkl").exists()
    assert (MODELS_DIR / "business_curve.csv").exists()
    assert (MODELS_DIR / "test_predictions.parquet").exists()

    snapshot = artefatos["snapshot"]
    assert snapshot["alvo"] == TARGET
    nomes = {candidato["modelo"] for candidato in snapshot["candidatos"]}
    assert (
        snapshot["modelo"] in nomes
    ), "O campeão precisa ser um dos candidatos avaliados."


def test_candidatos_expoem_params_para_tracking(artefatos: dict) -> None:
    """Cada candidato carrega seus hiperparâmetros para virar um run rastreável."""
    candidatos = artefatos["snapshot"]["candidatos"]
    assert candidatos, "Nenhum candidato avaliado."
    for candidato in candidatos:
        assert candidato["params"], f"Candidato sem params: {candidato['modelo']}"


def test_metricas_factuais_de_demanda_sao_sadias(artefatos: dict) -> None:
    teste = artefatos["snapshot"]["teste"]
    for chave in ("rmse", "mae", "r2", "uplift_pct"):
        assert chave in teste, f"Métrica ausente: {chave}"
    assert teste["rmse"] > 0 and teste["mae"] > 0
    assert -1.0 <= teste["r2"] <= 1.0


def test_honestidade_do_contrafactual_esta_explicita(artefatos: dict) -> None:
    snapshot = artefatos["snapshot"]
    assert "contrafactual" in snapshot["honestidade"].lower()
    assert (
        "rmse_media_global" in snapshot["baselines"]
    ), "Faltou o baseline ingênuo de calibração."


def test_artefatos_respeitam_o_guardrail_de_vazamento(artefatos: dict) -> None:
    cartao = artefatos["cartao"]
    features = set(cartao["features_numericas"]) | set(cartao["features_categoricas"])
    assert features == set(FEATURES_NUMERICAS + FEATURES_CATEGORICAS)
    assert TARGET not in features

    vetadas = set(cartao["colunas_vetadas_por_vazamento"])
    assert vetadas == set(colunas_com_vazamento())
    assert features.isdisjoint(
        vetadas
    ), f"Vazamento reentrou pelo cartão: {features & vetadas}"


def test_grade_de_desconto_fica_no_suporte_identificado() -> None:
    assert GRADE_DESCONTO.min() == 0.0
    assert (
        GRADE_DESCONTO.max() == 0.12
    )  # teto do desconto sem promoção (contrafactual identificado)


def test_curva_de_negocio_cobre_a_grade() -> None:
    curva = pd.read_csv(MODELS_DIR / "business_curve.csv")
    assert list(curva["desconto_pct"]) == [float(d) for d in GRADE_DESCONTO]
    for coluna in (
        "qtd_media_prevista",
        "receita_media_prevista",
        "lucro_medio_previsto",
    ):
        assert coluna in curva.columns
    assert (curva["qtd_media_prevista"] >= 0).all()


def test_site_publica_apenas_os_artefatos_publicos() -> None:
    """O showcase do Pages consome só os artefatos públicos — nunca o `.pkl`."""
    for nome in BUNDLE_DOCS:
        assert (DOCS_DIR / nome).exists(), f"Faltou no site: {nome}"
    assert not (DOCS_DIR / "best_model.pkl").exists(), "O modelo não vai para o site."
    assert "best_model.pkl" not in BUNDLE_DOCS
