from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

Instrumentator().instrument(app).expose(app)


@app.get('/')
def root():
    return {'message': 'FastAPI with Prometheus, Grafana & Docker!'}


@app.get('/ping')
def ping():
    return {'message': 'pong'}