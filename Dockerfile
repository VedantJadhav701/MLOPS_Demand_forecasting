FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Fix MLflow absolute paths (Windows -> Linux)
RUN find mlruns -name "meta.yaml" -exec sed -i 's|file:///C:/Users/HP/projects/MLOPS_demand_forecasting/mlruns|file:///app/mlruns|g' {} +

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
