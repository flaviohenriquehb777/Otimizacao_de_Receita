"""App de decisão de preço e desconto — Otimização de Receita (CRISP-DM: Deployment).

Simulador que usa o **modelo real** e a **lógica de negócio real** do pipeline
(`matriz_lucro`/`politica_otima`), respeitando o piso `preco_minimo` por item.
A UI é uma casca fina sobre `app.simulador`; os dados são de-identificados.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from app import simulador  # noqa: E402  (precisa da raiz no sys.path acima)

MESES = {
    1: "Jan",
    2: "Fev",
    3: "Mar",
    4: "Abr",
    5: "Mai",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Set",
    10: "Out",
    11: "Nov",
    12: "Dez",
}

st.set_page_config(
    page_title="Otimização de Receita — Preço & Desconto",
    page_icon="🍰",
    layout="wide",
)


@st.cache_resource(show_spinner="Carregando modelo e artefatos...")
def _bundle() -> dict:
    return simulador.carregar_bundle()


bundle = _bundle()
cartao = bundle["cartao"]
metricas = bundle["metricas"]
catalogo = bundle["catalogo"]
modelo = bundle["modelo"]
proveniencia = cartao.get("proveniencia", {})

st.title("Otimização de Receita — Preço & Desconto")
st.caption(
    "Modelo prescritivo que estima a demanda e escolhe o desconto que **maximiza "
    "o lucro esperado**, respeitando o preço mínimo de cada item. "
    f"Campeão: `{cartao['modelo']}` · commit `{proveniencia.get('commit', 'local')}` "
    f"· hash dos dados `{proveniencia.get('hash_dados', '—')}`."
)

# --- Cenário (sidebar) -------------------------------------------------------
loja = st.sidebar.selectbox("Unidade", catalogo["dominios"]["loja"])
produtos = sorted(t["produto"] for t in catalogo["templates"] if t["loja"] == loja)
produto = st.sidebar.selectbox("Produto", produtos)
template = next(
    t for t in catalogo["templates"] if t["loja"] == loja and t["produto"] == produto
)

st.sidebar.subheader("Alavancas de decisão")
preco_original = st.sidebar.number_input(
    "Preço de tabela (R$)",
    min_value=1.0,
    value=float(template["preco_original"]),
    step=1.0,
    format="%.2f",
)
custo_producao = st.sidebar.number_input(
    "Custo de produção (R$)",
    min_value=0.0,
    value=float(template["custo_producao"]),
    step=1.0,
    format="%.2f",
)
preco_minimo = st.sidebar.number_input(
    "Preço mínimo / piso (R$)",
    min_value=0.0,
    value=float(template["preco_minimo"]),
    step=1.0,
    format="%.2f",
)

with st.sidebar.expander("Contexto do dia"):
    mes = st.selectbox(
        "Mês", list(MESES), format_func=MESES.get, index=int(template["mes"]) - 1
    )
    fim_de_semana = st.toggle("Fim de semana", value=bool(template["fim_de_semana"]))
    promocao = st.toggle("Promoção ativa", value=bool(template["promocao_flag"]))

X = simulador.linha_de_cenario(
    template,
    preco_original=preco_original,
    custo_producao=custo_producao,
    preco_minimo=preco_minimo,
    mes=mes,
    fim_de_semana=int(fim_de_semana),
    promocao_flag=int(promocao),
)

if preco_minimo > preco_original:
    st.warning(
        "O preço mínimo está acima do preço de tabela: só o preço cheio é viável."
    )
if preco_original <= custo_producao:
    st.warning("O preço de tabela não cobre o custo de produção: lucro negativo.")

rec = simulador.recomendar(modelo, X)
curva = simulador.curva_do_cenario(modelo, X)

# --- Recomendação ------------------------------------------------------------
st.subheader("Recomendação para este cenário")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Desconto ótimo", f"{rec['desconto_otimo'] * 100:.0f}%")
col2.metric("Preço sugerido", f"R$ {rec['preco_otimo']:,.2f}")
col3.metric(
    "Lucro esperado / dia",
    f"R$ {rec['lucro_otimo']:,.2f}",
    delta=f"{rec['uplift_pct']:+.1f}% vs. preço cheio",
)
col4.metric("Lucro no preço cheio", f"R$ {rec['lucro_baseline']:,.2f}")

esq, dir_ = st.columns(2)
with esq:
    st.markdown("**Lucro esperado por desconto**")
    st.line_chart(curva, x="desconto_pct", y="lucro_esperado", height=320)
with dir_:
    st.markdown("**Demanda prevista por desconto**")
    st.line_chart(curva, x="desconto_pct", y="demanda_prevista", height=320)

st.caption(
    "Grade limitada a 12% = teto do desconto de balcão **sem promoção** — a única "
    "faixa em que o contrafactual está identificado nos dados. O lucro é uma "
    "estimativa contrafactual, não um resultado realizado."
)

with st.expander("Ver números da curva"):
    st.dataframe(curva, hide_index=True, width="stretch")

# --- Qualidade do modelo -----------------------------------------------------
st.divider()
st.subheader("Qualidade do modelo no teste (Q4/2025 — pico festivo)")
teste = metricas["teste"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("R² (demanda)", f"{teste['r2']:.3f}")
m2.metric("RMSE (demanda)", f"{teste['rmse']:.3f}")
m3.metric("MAE (demanda)", f"{teste['mae']:.3f}")
m4.metric("Uplift médio da política", f"{teste['uplift_pct']:.2f}%")
st.caption(metricas.get("honestidade", ""))

graf1, graf2 = st.columns(2)
with graf1:
    st.markdown("**Importância das features**")
    topo = sorted(
        cartao["importancia_features"].items(), key=lambda kv: kv[1], reverse=True
    )[:10]
    st.bar_chart(
        pd.DataFrame(topo, columns=["feature", "importancia"]).set_index("feature"),
        height=360,
    )
with graf2:
    st.markdown("**Curva de negócio (média da operação)**")
    st.line_chart(
        bundle["curva_negocio"], x="desconto_pct", y="lucro_medio_previsto", height=360
    )

with st.expander("Baselines ingênuos (calibração do ganho)"):
    st.json(metricas["baselines"])
