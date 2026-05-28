# Git Flow

[← Back to README](../../README.md)

## Contents

- [Branching model](#branching-model)
- [Pull request rules](#pull-request-rules)
- [Conventional Commits](#conventional-commits)

## Branching Model

The project uses a simple feature branch flow:

1. `main` is the stable branch.
2. Development happens in feature branches, for example `features` or
   `feature/<task-name>`.
3. A pull request is opened from the feature branch to `main`.
4. GitHub Actions must pass lint, formatting, tests, Docker build and deploy
   smoke checks.
5. After review and green CI, changes are merged into `main`.

## Pull Request Rules

- Keep changes focused.
- Update documentation when behavior or commands change.
- Add or update tests for API, repository and pipeline behavior.
- Do not commit local state, secrets, downloaded datasets or generated SQLite files.

## Conventional Commits

Use Conventional Commits:

```text
feat(api): add prediction history endpoint
fix(ci): validate kubernetes manifests
test: add drift calculator tests
docs: update minikube launch guide
chore(dvc): move credentials to environment
```

Common types:

| Type | Use |
| --- | --- |
| `feat` | User-visible feature |
| `fix` | Bug fix |
| `test` | Tests |
| `docs` | Documentation |
| `chore` | Tooling, maintenance |
| `ci` | CI/CD changes |
| `refactor` | Internal code changes without behavior change |
