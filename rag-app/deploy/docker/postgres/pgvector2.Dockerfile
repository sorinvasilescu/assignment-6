# Stage 1: Build pgvector
FROM postgres:alpine AS builder

# Install necessary packages including clang
RUN apk add --no-cache \
    build-base \
    git \
    postgresql-dev \
    clang \
    llvm-dev

WORKDIR /build

# Clone, compile, and install pgvector
RUN git clone --branch v0.5.0 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install

# Stage 2: Final image
FROM postgres:alpine

# Copy the compiled pgvector extension to the final image
COPY --from=builder /usr/local/lib/postgresql/ /usr/local/lib/postgresql/
COPY --from=builder /usr/local/share/postgresql/ /usr/local/share/postgresql/

ENV POSTGRES_USER=myuser
ENV POSTGRES_PASSWORD=mypassword
ENV POSTGRES_DB=mydb
