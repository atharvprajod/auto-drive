FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install HPC-specific packages
RUN pip3 install --no-cache-dir \
    dask[complete] \
    numpy \
    pandas \
    scipy \
    scikit-learn

# Set up workspace
WORKDIR /data
COPY . /data/

# Default command
CMD ["python3", "pipeline_main.py"] 