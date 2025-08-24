import time
import logging
from fastapi import FastAPI, Request

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] (line %(lineno)d) - %(levelname)s - %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S"
)
logger = logging.getLogger('profiler')

app = FastAPI()


@app.middleware('http')
async def add_timing(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    time_taken = time.time() - start_time
    logger.info(f"Request to {request.url.path} took {time_taken:.3f} seconds")
    return response


@app.get('/')
def home():
    return {'message': 'Profiling Demo'}


@app.get('/slow')
async def slow_endpoint():
    time.sleep(2)
    return {'message': 'This is a slow endpoint'}


@app.get('/fast')
async def fast_endpoint():
    return {'message': 'This is a fast endpoint'}