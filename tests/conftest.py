"""Fixtures compartilhadas da suíte.

O treino é o passo caro do projeto: roda **uma vez por sessão** e publica tanto o
contrato de artefatos (`models/`) quanto o bundle de serving do app
(`app/artifacts/`), que os testes reusam.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def treino_executado() -> None:
    """Executa o pipeline uma vez e publica artefatos + bundle do app."""
    from src.modeling.train_pipeline import main

    main()
