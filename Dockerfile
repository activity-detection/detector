FROM nvidia/cuda:13.1.2-base-ubuntu24.04

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

COPY uv.lock pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Lepiej nic nie zmieniać przed tym komentarzem bo obraz bedzie się długo tworzył
ENV YOLO_CONFIG_DIR=/
ENV BASE_MODEL_PATH=models/yolo26m_main.pt
ENV POSE_MODEL_PATH=models/yolo11m-pose.pt
ENV LSTM_MODEL_PATH=models/lstm_5class_1.pth
ENV LOG_CONFIG_PATH=logging_config/stdout.json

RUN mkdir Ultralytics

COPY models/final_models/ models/
COPY logging_config/ logging_config/

COPY main.py .
COPY src/ ./src

# USER appuser

CMD ["uv", "run", "--no-dev", "main.py"]