# Contributing

[← Back to README](README.md)

Thank you for improving the project. Keep contributions focused, documented and
covered by tests where possible.

## Workflow

1. Create or update a feature branch.
2. Make a focused change.
3. Update docs if commands, runtime behavior or architecture changed.
4. Run local checks.
5. Open a pull request to `main`.

## Local Checks

```bash
uv run flake8 src tests
uv run black --check src tests
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=60
```

Optional checks:

```bash
docker compose config --quiet
docker build -t predictive-maintenance-mlops:ci-check .
```

## Commit Style

Use Conventional Commits:

```text
feat(api): add prediction history endpoint
fix(ci): validate kubernetes manifests
test: add repository tests
docs: restructure documentation hub
```

## Do Not Commit

- `.env`
- secrets or credentials
- `state/predictions.db`
- raw downloaded datasets not tracked through DVC
- `.coverage`, `.pytest_cache/`, `htmlcov/`

## Documentation

Documentation lives in `docs/`. Add a link from [README.md](README.md) or
[docs/README.md](docs/README.md) when adding a new page.
