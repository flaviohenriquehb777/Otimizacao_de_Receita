"""Simulação de preço/desconto do app (lógica pura, sem Streamlit).

Reusa a **regra de negócio real** do pipeline de treino — o mesmo otimizador
(`matriz_lucro` / `politica_otima`) e a mesma coerência de features
(`aplicar_desconto`) — em vez de duplicá-la. Ficar separado da UI torna tudo
testável sem subir o Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config.paths import APP_ARTIFACTS_DIR
from src.modeling import train_pipeline as tp

FEATURES: list[str] = list(tp.FEATURES_NUMERICAS + tp.FEATURES_CATEGORICAS)
GRADE_DESCONTO: np.ndarray = tp.GRADE_DESCONTO


def carregar_bundle(diretorio: Path | None = None) -> dict[str, object]:
    """Carrega modelo + artefatos do bundle de serving (`app/artifacts/`)."""
    diretorio = Path(diretorio or APP_ARTIFACTS_DIR)
    esperados = (*tp.BUNDLE_APP, "app_catalog.json")
    faltantes = [nome for nome in esperados if not (diretorio / nome).exists()]
    if faltantes:
        raise FileNotFoundError(
            f"Bundle do app incompleto em {diretorio}: faltam {faltantes}. "
            "Rode 'python -m src.modeling.train_pipeline'."
        )

    def _json(nome: str) -> dict:
        return json.loads((diretorio / nome).read_text(encoding="utf-8"))

    return {
        "modelo": joblib.load(diretorio / "best_model.pkl"),
        "cartao": _json("model_card.json"),
        "metricas": _json("metrics_snapshot.json"),
        "catalogo": _json("app_catalog.json"),
        "curva_negocio": pd.read_csv(diretorio / "business_curve.csv"),
    }


def linha_de_cenario(template: dict, **ajustes: object) -> pd.DataFrame:
    """Monta a linha de features do cenário a partir do template + alavancas.

    Aceita qualquer feature como keyword (ex.: `preco_original`, `mes`) e
    restaura a coerência de preço no desconto zero. O otimizador reescreve
    `desconto_pct`/`preco_final`/`indice_preco_competidor` ao avaliar cada ponto
    da grade, então o app só precisa entregar uma linha semanticamente válida.
    """
    desconhecidas = set(ajustes) - set(FEATURES)
    if desconhecidas:
        raise KeyError(f"Features desconhecidas: {sorted(desconhecidas)}")

    linha = {**template, **ajustes}
    linha["desconto_pct"] = 0.0
    linha["preco_final"] = round(float(linha["preco_original"]), 2)
    linha["indice_preco_competidor"] = round(
        float(linha["preco_original"]) / float(linha["preco_competidor"]), 4
    )
    return pd.DataFrame([linha])[FEATURES]


def curva_do_cenario(
    modelo, X: pd.DataFrame, grade: np.ndarray | None = None
) -> pd.DataFrame:
    """Curva de demanda/receita/lucro por desconto para o cenário (uma linha)."""
    grade = GRADE_DESCONTO if grade is None else grade
    preco_cheio = float(X["preco_original"].iloc[0])
    custo = float(X["custo_producao"].iloc[0])

    linhas = []
    for desconto in grade:
        d = float(desconto)
        qtd = float(np.asarray(modelo.predict(tp.aplicar_desconto(X, d))).ravel()[0])
        preco = preco_cheio * (1.0 - d)
        linhas.append(
            {
                "desconto_pct": d,
                "preco_final": round(preco, 2),
                "demanda_prevista": qtd,
                "receita_esperada": preco * qtd,
                "lucro_esperado": (preco - custo) * qtd,
            }
        )
    return pd.DataFrame(linhas)


def recomendar(
    modelo, X: pd.DataFrame, grade: np.ndarray | None = None
) -> dict[str, float | bool]:
    """Desconto ótimo (respeitando `preco_minimo`) e ganho vs. preço de tabela."""
    grade = GRADE_DESCONTO if grade is None else grade
    desconto, lucro_politica, lucro_baseline = tp.politica_otima(modelo, X, grade)

    d_otimo = float(desconto[0])
    lucro_otimo = float(lucro_politica[0])
    lucro_cheio = float(lucro_baseline[0])
    preco_cheio = float(X["preco_original"].iloc[0])

    return {
        "desconto_otimo": d_otimo,
        "preco_otimo": round(preco_cheio * (1.0 - d_otimo), 2),
        "lucro_otimo": lucro_otimo,
        "lucro_baseline": lucro_cheio,
        "uplift_pct": (
            (lucro_otimo - lucro_cheio) / lucro_cheio * 100.0 if lucro_cheio else 0.0
        ),
        "viavel": bool(np.isfinite(lucro_otimo)),
    }
