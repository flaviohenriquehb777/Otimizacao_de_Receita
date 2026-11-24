"""Gerador do dataset sintético calibrado de vendas (confeitaria artesanal).

Motivação
---------
Os dados transacionais reais (POS/ERP, iFood, WhatsApp) não são públicos nem
foram fornecidos. Para que o projeto seja executável de ponta a ponta e para
provar o sinal de negócio (otimização de preço/desconto), geramos uma base
sintética **calibrada** por factos verificados do negócio:

- Tabela de preços real por linha/tamanho de produto.
- Mecânicas promocionais reais: "item do mês" e "dose dupla" (leve 2, pague 1).
- Duas unidades com perfis distintos (polo comercial x bairro residencial).
- Sazonalidade festiva, clima e calendário de feriados (Niterói, RJ).

De-identificação
----------------
Calibramos com números reais, porém o dataset público usa **nomenclatura
genérica** (unidades, produtos e bairros de-identificados), conforme a regra de
higiene de PII/marca do projeto. O mapeamento real -> genérico fica no dossiê
interno (não versionado publicamente).

Reprodutibilidade
-----------------
Determinístico com `SEED = 42`. Uma execução produz sempre o mesmo Parquet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import DATA_DIR

SEED = 42
DATA_INICIO = "2024-01-01"
DATA_FIM = "2025-12-31"
SAIDA_PARQUET = DATA_DIR / "dataset_cafeterias_rj.parquet"

# Âncora de receita: rede de 2 unidades com delivery forte, calibrada no teto de
# EPP do Simples Nacional (~R$ 4 mi/ano no total). Ajusta a escala de demanda
# para que a base sintética permaneça defensável face ao porte real do negócio.
ESCALA_DEMANDA = 0.71


# --- Calendário --------------------------------------------------------------
# Feriados nacionais brasileiros com efeito relevante sobre o consumo de festa.
_FERIADOS = {
    # 2024
    "2024-01-01", "2024-02-12", "2024-02-13", "2024-03-29", "2024-04-21",
    "2024-05-01", "2024-05-30", "2024-09-07", "2024-10-12", "2024-11-02",
    "2024-11-15", "2024-11-20", "2024-12-25",
    # 2025
    "2025-01-01", "2025-03-03", "2025-03-04", "2025-04-18", "2025-04-21",
    "2025-05-01", "2025-06-19", "2025-09-07", "2025-10-12", "2025-11-02",
    "2025-11-15", "2025-11-20", "2025-12-25",
}


@dataclass(frozen=True)
class Produto:
    """SKU de-identificado calibrado pela tabela de preços do negócio."""

    nome: str
    categoria: str
    tipo_produto: str
    tamanho_embalagem: str
    preco_original: float
    custo_ratio: float          # custo de produção como fração do preço
    elasticidade: float         # elasticidade-preço da demanda (negativa)
    base_demanda: float         # unidades/dia por unidade de referência


# Catálogo (preços reais convertidos para rótulos genéricos).
CATALOGO: tuple[Produto, ...] = (
    # Linha de tortas — três faixas e cinco tamanhos por faixa.
    Produto("Torta Standard 10 fatias", "Torta", "Torta", "10 fatias", 115.0, 0.35, -1.20, 3.2),
    Produto("Torta Standard 20 fatias", "Torta", "Torta", "20 fatias", 169.0, 0.35, -1.20, 2.1),
    Produto("Torta Standard 30 fatias", "Torta", "Torta", "30 fatias", 249.0, 0.35, -1.18, 1.4),
    Produto("Torta Standard 40 fatias", "Torta", "Torta", "40 fatias", 297.0, 0.35, -1.15, 0.9),
    Produto("Torta Standard 60 fatias", "Torta", "Torta", "60 fatias", 469.0, 0.35, -1.10, 0.4),
    Produto("Torta Premium 10 fatias", "Torta", "Torta", "10 fatias", 139.0, 0.38, -1.25, 2.0),
    Produto("Torta Premium 20 fatias", "Torta", "Torta", "20 fatias", 209.0, 0.38, -1.25, 1.3),
    Produto("Torta Premium 30 fatias", "Torta", "Torta", "30 fatias", 287.0, 0.38, -1.22, 0.9),
    Produto("Torta Premium 40 fatias", "Torta", "Torta", "40 fatias", 369.0, 0.38, -1.20, 0.6),
    Produto("Torta Premium 60 fatias", "Torta", "Torta", "60 fatias", 497.0, 0.38, -1.15, 0.3),
    Produto("Torta com Frutas 10 fatias", "Torta", "Torta", "10 fatias", 159.0, 0.42, -1.30, 1.8),
    Produto("Torta com Frutas 20 fatias", "Torta", "Torta", "20 fatias", 219.0, 0.42, -1.30, 1.2),
    Produto("Torta com Frutas 30 fatias", "Torta", "Torta", "30 fatias", 297.0, 0.42, -1.28, 0.8),
    Produto("Torta com Frutas 40 fatias", "Torta", "Torta", "40 fatias", 369.0, 0.42, -1.25, 0.5),
    Produto("Torta com Frutas 60 fatias", "Torta", "Torta", "60 fatias", 519.0, 0.42, -1.20, 0.25),
    # Consumo imediato.
    Produto("Mini Cake Individual", "Torta", "Mini Cake", "individual", 49.9, 0.32, -1.05, 6.0),
    Produto("Fatia de Torta", "Torta", "Fatia", "individual", 24.0, 0.34, -1.40, 9.0),
    # Linha salgada.
    Produto("Empadao Congelado 1,2 kg", "Salgado", "Empadao", "1,2 kg", 98.0, 0.40, -1.10, 1.1),
    Produto("Quiche Base 500 g", "Salgado", "Quiche", "500 g", 38.0, 0.38, -1.15, 2.2),
    Produto("Salgadinho Frito (cento)", "Salgado", "Salgadinho", "cento", 129.0, 0.42, -1.12, 1.6),
    Produto("Salgadinho Assado (cento)", "Salgado", "Salgadinho", "cento", 189.0, 0.42, -1.12, 1.0),
    # Doces de festa e kits.
    Produto("Docinho de Festa (cento)", "Doce", "Docinho", "cento", 240.0, 0.30, -1.30, 1.2),
    Produto("Kit Festa 2", "Kit", "Kit", "kit", 379.0, 0.36, -0.95, 0.7),
    Produto("Kit Docura 3", "Kit", "Kit", "kit", 389.0, 0.34, -0.95, 0.5),
)


@dataclass(frozen=True)
class Loja:
    """Unidade de-identificada com perfil operacional distinto."""

    nome: str
    regiao: str
    bairro: str
    perfil_loja: str
    fator_demanda: float                  # escala de demanda relativa
    mix_canal: tuple[tuple[str, float], ...]  # canal -> probabilidade


LOJAS: tuple[Loja, ...] = (
    Loja(
        "Unidade A", "Regiao A", "Bairro A", "Shopping",
        fator_demanda=1.35,
        mix_canal=(("Delivery", 0.45), ("Balcao", 0.35), ("Mesa", 0.20)),
    ),
    Loja(
        "Unidade B", "Regiao B", "Bairro B", "Rua",
        fator_demanda=1.00,
        mix_canal=(("Balcao", 0.40), ("Delivery", 0.35), ("Mesa", 0.25)),
    ),
)

# Clima médio mensal (litoral sudeste): (temperatura media, chuva media em mm).
_CLIMA_MENSAL = {
    1: (27.5, 5.5), 2: (28.0, 4.5), 3: (27.0, 5.0), 4: (25.5, 4.0),
    5: (23.5, 3.2), 6: (22.0, 2.2), 7: (21.5, 2.0), 8: (22.0, 2.0),
    9: (22.5, 3.0), 10: (23.5, 4.0), 11: (24.5, 4.8), 12: (26.5, 5.5),
}

# Sazonalidade festiva (multiplicador de demanda por mês).
_SAZONALIDADE_MES = {
    1: 0.95, 2: 0.95, 3: 1.00, 4: 1.05, 5: 1.15, 6: 1.10,
    7: 1.00, 8: 1.05, 9: 0.95, 10: 1.10, 11: 1.10, 12: 1.35,
}

# Multiplicador por dia da semana (0 = segunda).
_SAZONALIDADE_DIA = {0: 0.90, 1: 0.90, 2: 0.95, 3: 1.00, 4: 1.20, 5: 1.40, 6: 1.25}

# Eventos sazonais de-identificados (mês -> rótulo do evento).
_EVENTOS_MES = {
    4: "Pascoa", 5: "Dia das Maes", 6: "Festa Junina", 12: "Festas de Fim de Ano",
}


def _descontos_item_do_mes() -> dict[str, float]:
    """Descontos do "item do mês" por tamanho (mecânica real calibrada)."""
    return {"10 fatias": 0.148, "20 fatias": 0.059, "30 fatias": 0.120}


_LINHAS_TORTA = ("Torta Standard", "Torta Premium", "Torta com Frutas")


def _linha_do_mes(data: pd.Timestamp) -> str:
    """Rotaciona, de forma determinística, a linha de torta promovida no mês."""
    idx = (data.year * 12 + data.month) % len(_LINHAS_TORTA)
    return _LINHAS_TORTA[idx]


def _gerar_calendario() -> pd.DataFrame:
    """Gera o calendário diário com clima, feriados e eventos."""
    datas = pd.date_range(DATA_INICIO, DATA_FIM, freq="D")
    rng = np.random.default_rng(SEED)
    temp, chuva = [], []
    for ts in datas:
        t_media, c_media = _CLIMA_MENSAL[ts.month]
        temp.append(rng.normal(t_media, 1.8))
        chuva.append(max(0.0, rng.gamma(shape=1.2, scale=c_media / 1.2)))
    cal = pd.DataFrame(
        {
            "data": datas,
            "dia_semana_idx": datas.dayofweek,
            "mes": datas.month,
            "ano": datas.year,
            "temperatura": np.round(temp, 1),
            "chuva_mm": np.round(chuva, 1),
        }
    )
    cal["feriado_flag"] = cal["data"].dt.strftime("%Y-%m-%d").isin(_FERIADOS).astype(int)
    cal["evento_especial"] = cal["mes"].map(_EVENTOS_MES)
    cal["sazonal"] = cal["mes"].map(_SAZONALIDADE_MES) * cal["dia_semana_idx"].map(
        _SAZONALIDADE_DIA
    )
    # Clima: dia mais frio favorece doces; chuva forte reduz tráfego de rua.
    cal["clima_lift"] = 1.0 + (0.02 * (24.0 - cal["temperatura"])).clip(-0.15, 0.15)
    cal["clima_lift"] *= 1.0 - (0.01 * cal["chuva_mm"]).clip(0.0, 0.20)
    return cal


def _promocao_ativa(data: pd.Timestamp, loja: Loja, produto: Produto) -> str | None:
    """Retorna o tipo de promoção vigente para (data, loja, produto)."""
    descontos = _descontos_item_do_mes()
    # "Dose dupla": leve 2, pague 1 — apenas na fatia, em quintas específicas.
    if produto.tipo_produto == "Fatia":
        # Penúltima quinta do mês na Unidade B; última quinta na Unidade A.
        if data.dayofweek == 3:
            quintas = pd.date_range(
                data.replace(day=1), data + pd.offsets.MonthEnd(0), freq="W-THU"
            )
            if loja.nome == "Unidade A" and len(quintas) and data == quintas[-1]:
                return "Dose Dupla"
            if loja.nome == "Unidade B" and len(quintas) >= 2 and data == quintas[-2]:
                return "Dose Dupla"
    # "Item do mes": desconto mensal em uma unica linha de torta (rotativa).
    if (
        produto.categoria == "Torta"
        and produto.tamanho_embalagem in descontos
        and produto.nome.startswith(_linha_do_mes(data))
    ):
        return "Item do Mes"
    return None


def _desconto_efetivo(promo: str | None, produto: Produto, rng: np.random.Generator) -> float:
    """Percentual de desconto aplicado ao preço de tabela."""
    if promo == "Dose Dupla":
        return 0.50  # equivalente a leve 2, pague 1
    if promo == "Item do Mes":
        base = _descontos_item_do_mes()[produto.tamanho_embalagem]
        return float(np.clip(base + rng.normal(0.0, 0.005), 0.0, 0.60))
    # Descontos residuais de balcão/canal (pequenos e frequentes).
    if rng.random() < 0.10:
        return float(np.clip(rng.uniform(0.03, 0.12), 0.0, 0.60))
    return 0.0


def gerar_dataset() -> pd.DataFrame:
    """Gera o dataset sintético calibrado (grão: data x loja x produto)."""
    rng = np.random.default_rng(SEED)
    cal = _gerar_calendario()

    registros: list[dict[str, object]] = []
    for loja in LOJAS:
        canais, probs = zip(*loja.mix_canal)
        for produto in CATALOGO:
            preco_ref = produto.preco_original
            base = produto.base_demanda * loja.fator_demanda * ESCALA_DEMANDA
            for linha in cal.itertuples(index=False):
                data = pd.Timestamp(linha.data)
                promo = _promocao_ativa(data, loja, produto)
                # Arredonda o desconto ANTES de precificar para que o preço
                # publicado seja exatamente reconstruível a partir das colunas
                # gravadas (`preco_original`, `desconto_pct`): evita divergência
                # por arredondamento duplo e mantém a base auto-consistente.
                desconto = round(_desconto_efetivo(promo, produto, rng), 4)
                preco_final = round(round(preco_ref, 2) * (1.0 - desconto), 2)

                # Demanda: elasticidade-preço x sazonalidade x clima x promo.
                lift_preco = (preco_final / preco_ref) ** produto.elasticidade
                lift_promo = 1.75 if promo == "Dose Dupla" else (1.30 if promo else 1.0)
                ruido = float(rng.lognormal(mean=0.0, sigma=0.18))
                demanda = base * lift_preco * linha.sazonal * linha.clima_lift * lift_promo * ruido
                if linha.feriado_flag:
                    demanda *= 1.12

                estoque = int(max(0, round(demanda * rng.uniform(0.85, 1.65))))
                quantidade = int(round(demanda))
                ruptura = int(estoque < quantidade)
                if ruptura:
                    quantidade = int(round(estoque * 0.9))
                quantidade = max(quantidade, 0)

                custo = round(preco_ref * produto.custo_ratio * rng.uniform(0.95, 1.05), 2)
                preco_competidor = round(preco_ref * rng.uniform(0.90, 1.16), 2)
                margem_minima = float(np.round(rng.uniform(0.25, 0.40), 3))
                gasto_marketing = round(
                    rng.uniform(400.0, 1200.0) * (1.6 if promo else 1.0), 2
                )
                custo_logistico = round(
                    rng.uniform(4.0, 9.0) if loja.perfil_loja == "Shopping" else rng.uniform(3.0, 7.0),
                    2,
                )
                registros.append(
                    {
                        "data": data,
                        "loja": loja.nome,
                        "produto": produto.nome,
                        "categoria": produto.categoria,
                        "custo_producao": custo,
                        "preco_original": round(preco_ref, 2),
                        "desconto_pct": round(desconto, 4),
                        "preco_final": preco_final,
                        "quantidade_vendida_dia": quantidade,
                        "dia_semana": data.day_name(),
                        "mes": int(linha.mes),
                        "feriado_flag": int(linha.feriado_flag),
                        "evento_especial": linha.evento_especial,
                        "promocao_flag": int(promo is not None),
                        "promocao_tipo": promo,
                        "promocao_inicio": (
                            data.replace(day=1) if promo == "Item do Mes"
                            else (data if promo else pd.NaT)
                        ),
                        "promocao_fim": (
                            data + pd.offsets.MonthEnd(0) if promo == "Item do Mes"
                            else (data if promo else pd.NaT)
                        ),
                        "canal": str(rng.choice(canais, p=probs)),
                        "regiao": loja.regiao,
                        "bairro": loja.bairro,
                        "perfil_loja": loja.perfil_loja,
                        "marca": "Artesanal",
                        "tamanho_embalagem": produto.tamanho_embalagem,
                        "tipo_produto": produto.tipo_produto,
                        "preco_competidor": preco_competidor,
                        "indice_preco_competidor": round(preco_final / preco_competidor, 4),
                        "gasto_marketing": gasto_marketing,
                        "campanha_id": (
                            f"CAMP-{data.year}-{data.month:02d}" if promo else None
                        ),
                        "estoque_disponivel": estoque,
                        "ruptura_flag": ruptura,
                        "lead_time_reposicao": int(rng.integers(1, 6)),
                        "temperatura": float(linha.temperatura),
                        "chuva_mm": float(linha.chuva_mm),
                        "custo_logistico": custo_logistico,
                        "custo_marketing_unitario": round(
                            gasto_marketing / max(quantidade, 1), 2
                        ),
                        "preco_minimo": round(custo * (1.0 + margem_minima), 2),
                        "margem_minima": margem_minima,
                    }
                )

    df = pd.DataFrame.from_records(registros)
    # Volume mensal observado por (mes, loja, produto) — alimenta o alvo de receita.
    df["_mes_ref"] = df["data"].dt.to_period("M")
    df["quantidade_vendida_mes"] = (
        df.groupby(["_mes_ref", "loja", "produto"])["quantidade_vendida_dia"]
        .transform("sum")
        .astype("int64")
    )
    df = df.drop(columns="_mes_ref")
    df = df.sort_values(["data", "loja", "produto"]).reset_index(drop=True)
    return df


def salvar_parquet(df: pd.DataFrame, destino: Path = SAIDA_PARQUET) -> Path:
    """Persiste o dataset em Parquet (formato eficiente e rastreável)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)
    return destino


def main() -> None:
    """Gera e persiste o dataset sintético calibrado."""
    df = gerar_dataset()
    destino = salvar_parquet(df)
    print(f"Dataset gerado: {destino}")
    print(f"Linhas: {len(df):,} | Colunas: {df.shape[1]}")
    print(f"Periodo: {df['data'].min().date()} a {df['data'].max().date()}")
    print(f"Receita total simulada: R$ {(df['preco_final'] * df['quantidade_vendida_dia']).sum():,.2f}")


if __name__ == "__main__":
    main()
