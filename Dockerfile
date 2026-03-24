FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Fix MLflow absolute paths (Windows -> Linux)
# Fix MLflow metadata paths to point to the container's absolute path
# This handles both local dev paths and GitHub runner paths
RUN if [ -d "mlruns" ]; then \
    find mlruns -name "meta.yaml" -exec sed -i 's|file:///.*/mlruns|file:///app/mlruns|g' {} +; \
    else echo "WARNING: mlruns not found during build"; \
    fi

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
