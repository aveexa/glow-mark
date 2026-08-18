FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gcs_utils.py geometry_cols.py suggestion_model.py suggestion_rules.py suggestion_serve.py main.py ./

ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
