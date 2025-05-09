FROM python:3.12


WORKDIR /app

COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY infer.py .
COPY ./preprocessing ./preprocessing
COPY ./feature_extraction ./feature_extraction
COPY audio.py .
COPY export.py .
COPY client_id_gender_age_model_union.pkl .

ENTRYPOINT ["python", "infer.py","/data","/results"]
