# Quant Research Lab — Week 1 Starter

This repo bootstraps your Week 1: Dev environment, Git/GitHub flow, NumPy vectorization, and linear algebra refresher.

## Quickstart

```bash
# choose one: venv (built-in) or conda
# --- venv ---
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .[dev]

# run checks
ruff check .
black --check .
mypy src
pytest -q
```

## What’s here
- `src/qrutils/linalg.py` — LA utils (norms, dot, cosine similarity matrix, pairwise distances).
- `scripts/bench_vectorization.py` — micro-benchmarks comparing loops vs vectorization.
- `tests/test_linalg.py` — unit tests.
- GitHub Actions CI in `.github/workflows/ci.yml`.
- Tooling: black, ruff, mypy, pytest; configs in `pyproject.toml` and `mypy.ini`.

## Suggested Git flow
```bash
git init
git add .
git commit -m "chore: week1 starter scaffold"
# create a GitHub repo named quant-research-lab (or your choice), then:
git remote add origin YOUR_REPO_URL
git branch -M main
git push -u origin main
```
