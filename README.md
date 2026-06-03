
## Run 
```bash
uv sync
uv run panel serve src/app.py --dev --show
```

## Local checks

```bash
uv run ruff check src tests
uv run pytest
```

## Version control story

Use GitHub with pull requests:

1. Create a feature branch.
2. Open a pull request.
3. GitHub Actions runs linting and tests.
4. Merge only when checks pass.
5. A deployment job can run after merge to `main`.

This repo includes `.github/workflows/ci.yml` with:

- linting via Ruff
- tests via Pytest
- a dummy deployment stage showing where a real deployment command would go
