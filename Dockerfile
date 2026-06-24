FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.*+cpu torchvision==0.*+cpu 2>/dev/null || \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic pillow scikit-learn matplotlib python-multipart

COPY src src
COPY checkpoints checkpoints
COPY deployments/app_metrics.json app_metrics.json
ENV METRICS_PATH=/app/app_metrics.json
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
