FROM python:3.12-slim

WORKDIR /app

COPY cloud_api/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --timeout=300 --retries=10 -r requirements.txt

COPY cloud_api/app ./app

RUN mkdir -p /app/model

# Production model
ENV MODEL_URL="https://github.com/Spidy131/mlflow-fraud-detection/releases/download/v1.0.0/model.ubj"
ENV MODEL_SHA256="D97CC8A3AE85501B7356B527B86FBB7DBFDD544F014714569C9B84E3C7E0043E"

# Download production model
RUN python -c "import os, urllib.request; url=os.environ['MODEL_URL']; urllib.request.urlretrieve(url, '/app/model/model.ubj')"

# Verify production model checksum
RUN python -c "import hashlib, os; p='/app/model/model.ubj'; expected=os.environ['MODEL_SHA256'].lower(); actual=hashlib.sha256(open(p,'rb').read()).hexdigest().lower(); assert actual == expected, f'Checksum mismatch: expected={expected}, actual={actual}'; print('Model SHA256 verified:', actual)"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
