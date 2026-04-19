# Берём официальный образ Python версии 3.12 с минимальным размером
FROM python:3.12

# Устанавливаем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Создаём внутри контейнера папку /app и переходим в неё
WORKDIR /app

# Копируем файл с зависимостями из проекта в контейнер
COPY pyproject.toml uv.lock ./

# Устанавливаем все библиотеки из pyproject.toml
RUN uv sync --frozen --no-dev

# Копируем весь код и модель API сервиса в контейнер
COPY src/ ./src/

# Копируем папку с обученной моделью в контейнер
COPY models/ ./models/

# Активируем виртуальное окружение для CMD
ENV PATH="/app/.venv/bin:$PATH"

# Указываем, что контейнер будет слушать порт 8080
EXPOSE 8080

# Команда для запуска сервера при старте контейнера
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]