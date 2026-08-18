FROM python:3.11-slim AS build

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && curl -LsSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

RUN git config --global url."https://github.com/".insteadOf "git@github.com:" \
    && git clone --recursive https://github.com/chux0519/embeddings.cpp.git /src

RUN uv pip install --system ".[server]"

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    EMBEDDINGS_CPP_CPU_REPACK=1 \
    EMBEDDINGS_CPP_FLASH_ATTN=1

WORKDIR /app
COPY --from=build /usr/local /usr/local

EXPOSE 8080
ENTRYPOINT ["python", "-m", "embeddings_cpp.server"]
