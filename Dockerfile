FROM python:3.12-slim

# Install OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash \
    sudo \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create devcontainer user
RUN groupadd -r aas-rail && \
    useradd -r -g aas-rail -m -s /bin/bash aas-rail

# Give sudo privileges to the aas-rail user
RUN echo "aas-rail ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/aas-rail && \
    chmod 0440 /etc/sudoers.d/aas-rail

SHELL ["/bin/bash", "-c"]

# Switch to the application user
USER aas-rail
WORKDIR /home/aas-rail

# Create virtual environment
RUN python -m venv .venv
ENV PATH="/home/aas-rail/.venv/bin:$PATH"

# Install aas-rail and the development extras from pyproject.toml
WORKDIR /home/aas-rail/aas-rail
COPY --chown=aas-rail:aas-rail pyproject.toml README.md LICENSE ./
COPY --chown=aas-rail:aas-rail src ./src
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --editable ".[app,dev,perturbation]"

# Git safety configuration
RUN git config --global --add safe.directory /home/aas-rail/aas-rail

ARG GIT_USER_NAME
ARG GIT_USER_EMAIL
RUN git config --global user.name "${GIT_USER_NAME}" \
 && git config --global user.email "${GIT_USER_EMAIL}"


CMD ["sleep", "infinity"]
