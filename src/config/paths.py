"""Caminhos canônicos do projeto — fonte única de verdade.

Todos os módulos e notebooks devem importar daqui em vez de montar
caminhos relativos manualmente.
"""

from __future__ import annotations

from pathlib import Path

_MARCADORES = (".git", "pyproject.toml", "src")


def get_project_root() -> Path:
    """Retorna a raiz do projeto subindo até encontrar um marcador conhecido."""
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marcador).exists() for marcador in _MARCADORES):
            return candidate
    raise RuntimeError("Raiz do projeto não encontrada.")


PROJECT_ROOT = get_project_root()

DATA_DIR = PROJECT_ROOT / "dados"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "report"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# App de decisão (CRISP-DM — Deployment). O `artifacts/` é o **bundle de serving**
# enxuto e versionado no Git, consumido pelo Streamlit Cloud sem DVC nem a base
# crua. Os artefatos pesados de treino continuam sob governança do DVC em
# `models/`; aqui fica apenas a cópia pequena e publicável.
APP_DIR = PROJECT_ROOT / "app"
APP_ARTIFACTS_DIR = APP_DIR / "artifacts"

for _dir in (DATA_DIR, MODELS_DIR, REPORTS_DIR, APP_ARTIFACTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# --- Datasets (dados crus e tratados) ---------------------------------------
# Nomes de-identificados; o dono real da base é mantido fora do repositório
# público. O dataset canônico é gerado em Parquet e versionado via DVC.
DADOS_CAFETERIAS_PARQUET = DATA_DIR / "dataset_cafeterias_rj.parquet"
DADOS_CAFETERIAS_PREPARADO = DATA_DIR / "dataset_cafeterias_rj_preparado.parquet"
DADOS_AMOR_A_CAKES = DATA_DIR / "dataset_cafeterias_rj.xlsx"
DADOS_AMOR_A_CAKES_TRATADOS = DATA_DIR / "dataset_cafeterias_rj_limpo.xlsx"


def caminho_dataset() -> Path:
    """Retorna o dataset preferido: Parquet (DVC) e, na ausência, o XLSX legado."""
    if DADOS_CAFETERIAS_PARQUET.exists():
        return DADOS_CAFETERIAS_PARQUET
    return DADOS_AMOR_A_CAKES


def dataset_exists(caminho: Path | None = None) -> bool:
    """Verifica de forma preguiçosa se o dataset local existe.

    Evita quebrar import em ambientes CI; o pipeline decide o fallback
    para dados sintéticos quando o arquivo estiver ausente.
    """
    return (caminho or caminho_dataset()).exists()
