# The sandbox model-written code runs in for the Aider polyglot suite: Python + pytest, Rust + cargo,
# Node 22 with the Exercism JavaScript track's dev dependencies preinstalled. Tests run in it with
# --network none, so every dependency is baked here, never fetched at test time.
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytest==8.3.3
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.89.0 && chmod -R a+rwX /opt/cargo /opt/rustup
# Every crate the Rust track's manifests name (the stubs' own and the reference solutions'
# Cargo-example.toml), fetched at build time so `cargo --offline` resolves them in the sandbox.
WORKDIR /opt/cargo-prefetch
COPY cargo-prefetch.toml Cargo.toml
RUN mkdir src && touch src/lib.rs && cargo fetch && rm -rf src target && chmod -R a+rwX /opt/cargo
WORKDIR /opt/js
COPY js-package.json package.json
COPY js-babel.config.js babel.config.js
RUN npm install --no-audit --no-fund && chmod -R a+rX /opt/js
ENV NODE_PATH=/opt/js/node_modules
RUN useradd -m -u 10001 runner
USER runner
WORKDIR /work
