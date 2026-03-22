FROM python:3.13
WORKDIR /usr/local/app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY src ./src
COPY main.py ./
COPY .env ./

COPY vectors/ ./vectors/
COPY data/ ./data/
COPY models/saved-models/ ./models/saved-models/

EXPOSE 9090

CMD ["uv", "run", "main.py"]