Este diretório armazena o dataset canônico em **Parquet**, **não versionado** no
Git — a governança de dados é do **DVC** (ver `dvc.yaml` / `dvc.lock`).

Arquivos gerados pelo pipeline:
- `dataset_cafeterias_rj.parquet`: base bruta calibrada (`src.data.gerar_dataset`).
- `dataset_cafeterias_rj_preparado.parquet`: base preparada, com features
  leakage-safe e split temporal (`src.data.preparar_dataset`).

Reprodução determinística (`SEED = 42`):

    uv run dvc repro

Boas práticas:
- Não versionar dados no Git.
- Transportar/versionar os dados via DVC (remote remoto), nunca no repositório.
