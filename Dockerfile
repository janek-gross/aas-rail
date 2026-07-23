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

# Install Python dependencies
COPY --chown=aas-rail:aas-rail requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Git safety configuration
ENV PYTHONPATH=/home/aas-rail/vamos_evaluation_framework:$PYTHONPATH
RUN git config --global --add safe.directory /home/aas-rail/aas-rail
RUN git config --global --add safe.directory /home/aas-rail/vamos_evaluation_framework

ARG GIT_USER_NAME
ARG GIT_USER_EMAIL
RUN git config --global user.name "${GIT_USER_NAME}" \
 && git config --global user.email "${GIT_USER_EMAIL}"


CMD ["sleep", "infinity"]
