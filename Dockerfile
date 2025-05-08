FROM python:3.10


WORKDIR /app

COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY export_pipeline_cloudpickle.pkl .
COPY infer.py .
COPY ./preprocessing ./preprocessing
COPY ./feature_extraction ./feature_extraction
COPY audio.py .
COPY export.py .

