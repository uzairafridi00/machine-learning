FROM python:3.10

WORKDIR /app

COPY ./app /app

RUN pip install fastapi uvicorn prometheus-fastapi-instrumentator

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]