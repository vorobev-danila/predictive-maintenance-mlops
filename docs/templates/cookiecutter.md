# Cookiecutter Template

[← Back to README](../../README.md)

The repository includes a reusable template in:

```text
cookiecutter-template/
```

Generate a new project:

```bash
uvx cookiecutter cookiecutter-template
```

The template contains starting points for:

- FastAPI service;
- MLflow training pipeline;
- Docker and Docker Compose;
- Kubernetes manifests;
- tests;
- GitHub Actions.

Template configuration:

```text
cookiecutter-template/cookiecutter.json
```

Generated project root:

```text
cookiecutter-template/{{cookiecutter.project_slug}}/
```
