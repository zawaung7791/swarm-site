# ---- build the Rust lane as WebAssembly ----
FROM rust:1-slim-bookworm AS build
RUN rustup target add wasm32-unknown-unknown
WORKDIR /src
COPY rust/ ./
# SIMD lets Rust process four particles per instruction
ENV RUSTFLAGS="-C target-feature=+simd128"
RUN cargo build --release --target wasm32-unknown-unknown

# ---- serve everything as a static site ----
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY web/ /usr/share/nginx/html/
COPY --from=build /src/target/wasm32-unknown-unknown/release/swarm.wasm /usr/share/nginx/html/swarm.wasm
