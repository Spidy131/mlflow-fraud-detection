FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .

# Install all dependencies except XGBoost
RUN grep -v "^xgboost==" requirements-docker.txt > requirements-no-xgboost.txt && \
    pip install --no-cache-dir --timeout=300 --retries=10 -r requirements-no-xgboost.txt

# Install XGBoost without its optional GPU/NCCL dependency
RUN pip install --no-cache-dir --timeout=300 --retries=10 --no-deps xgboost==3.4.1

COPY app ./app

COPY models/fraud_model.skops ./models/fraud_model.skops
COPY models/threshold.txt ./models/threshold.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]