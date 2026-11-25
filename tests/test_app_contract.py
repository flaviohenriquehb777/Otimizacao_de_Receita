"""Contrato do app de serving (CRISP-DM — Deployment).

Verifica que o bundle publicado em `app/artifacts/` basta para o app — catálogo
leakage-safe, cenário montável e otimizador respeitando o piso de preço — sem
depender da base crua nem do DVC.
"""

import pytest

from app import simulador
from src.config.paths import APP_ARTIFACTS_DIR, PROJECT_ROOT
from src.data.preparar_dataset import colunas_com_vazamento
from src.modeling.train_pipeline import (
    BUNDLE_APP,
    FEATURES_CATEGORICAS,
    FEATURES_NUMERICAS,
    GRADE_DESCONTO,
    TARGET,
)


@pytest.fixture(scope="module")
def bundle(treino_executado: None) -> dict:
    """Carrega o bundle de serving publicado pelo pipeline."""
    return simulador.carregar_bundle()


def test_bundle_de_serving_esta_completo() -> None:
    for nome in (*BUNDLE_APP, "app_catalog.json"):
        assert (APP_ARTIFACTS_DIR / nome).exists(), f"Faltou no bundle: {nome}"


def test_catalogo_e_leakage_safe(bundle: dict) -> None:
    catalogo = bundle["catalogo"]
    features = set(catalogo["features_numericas"]) | set(
        catalogo["features_categoricas"]
    )
    assert features == set(FEATURES_NUMERICAS + FEATURES_CATEGORICAS)
    assert TARGET not in features
    assert features.isdisjoint(set(colunas_com_vazamento()))
    assert set(catalogo["dominios"]) == set(FEATURES_CATEGORICAS)


def test_templates_cobrem_todas_as_features_e_pares(bundle: dict) -> None:
    catalogo = bundle["catalogo"]
    esperado = set(FEATURES_NUMERICAS + FEATURES_CATEGORICAS)
    templates = catalogo["templates"]
    assert templates, "Catálogo sem cenários-tipo."
    pares = {(t["loja"], t["produto"]) for t in templates}
    assert len(pares) == len(templates), "Cenários-tipo duplicados por (loja, produto)."
    for template in templates:
        assert set(template) == esperado, "Template fora do contrato de features."


def test_recomendacao_respeita_o_piso_de_preco(bundle: dict) -> None:
    template = bundle["catalogo"]["templates"][0]
    X = simulador.linha_de_cenario(
        template, preco_original=40.0, custo_producao=12.0, preco_minimo=36.0
    )
    rec = simulador.recomendar(bundle["modelo"], X)
    # 40 * (1 - d) >= 36  =>  d <= 0.10 (o ótimo não pode furar o piso).
    assert rec["viavel"]
    assert rec["desconto_otimo"] <= 0.10 + 1e-9
    assert rec["preco_otimo"] >= 36.0 - 1e-9


def test_curva_cobre_a_grade(bundle: dict) -> None:
    template = bundle["catalogo"]["templates"][0]
    curva = simulador.curva_do_cenario(
        bundle["modelo"], simulador.linha_de_cenario(template)
    )
    assert list(curva["desconto_pct"]) == [float(d) for d in GRADE_DESCONTO]
    assert (curva["demanda_prevista"] >= 0).all()
    assert curva.iloc[0]["desconto_pct"] == 0.0


def test_linha_de_cenario_rejeita_feature_desconhecida(bundle: dict) -> None:
    template = bundle["catalogo"]["templates"][0]
    with pytest.raises(KeyError):
        simulador.linha_de_cenario(template, margem_unitaria=1.0)


def test_app_streamlit_renderiza_sem_erro(bundle: dict) -> None:
    """Fumaça da UI: o app sobe, recomenda e não emite avisos de negócio."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=90
    )
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value.startswith("Otimização de Receita")
    assert len(at.metric) >= 4
    assert not at.warning
