FROM python:{{ cookiecutter.python_version }}-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

COPY models/ ./models/

EXPOSE {{ cookiecutter.fastapi_port }}

CMD ["uvicorn", "{{ cookiecutter.package_name }}.api.main:app", "--host", "0.0.0.0", "--port", "{{ cookiecutter.fastapi_port }}"]
