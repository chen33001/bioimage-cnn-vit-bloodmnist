# ------------------------------------------------------------
# 🧠 Bioimage CNN vs Vision Transformer on BloodMNIST
# Python 3.12 + PyTorch + MedMNIST + Jupyter
# ------------------------------------------------------------

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Create working directory
WORKDIR /app

# Copy requirement files first (for caching)
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential \
 && pip install --upgrade pip \
 && pip install -r requirements.txt \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files into container
COPY . /app

# Expose port for Jupyter
EXPOSE 8888

# Default command: start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
