#FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Install AWS CLI for aarch64, refer to docs for more info https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && \
    unzip -o awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install git and allow for git push/pull from devcontainer.
RUN apt-get install git && \
    git config --global --add safe.directory /workspaces/oxford-genai-capstone

# Install python dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Final cleanup
RUN rm -rf /var/lib/apt/lists/*s
