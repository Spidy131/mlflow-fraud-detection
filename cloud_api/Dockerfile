FROM python:3.12-slim

WORKDIR /app

COPY cloud_api/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --timeout=300 --retries=10 -r requirements.txt

COPY cloud_api/app ./app

RUN mkdir -p /app/model

# Download production model from GitHub Release v1.0.0
ARG MODEL_URL
RUN python -c "import os, urllib.request; url=os.environ.get('MODEL_URL'); assert url, 'MODEL_URL build argument is required'; urllib.request.urlretrieve(url, '/app/model/model.ubj')"

# Verify production model checksum
ARG MODEL_SHA256
RUN python -c "import hashlib, os; p='/app/model/model.ubj'; expected=os.environ.get('MODEL_SHA256','').lower(); actual=hashlib.sha256(open(p,'rb').read()).hexdigest().lower(); assert expected and actual == expected, f'Checksum mismatch: expected={expected}, actual={actual}'; print('Model SHA256 verified:', actual)"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
