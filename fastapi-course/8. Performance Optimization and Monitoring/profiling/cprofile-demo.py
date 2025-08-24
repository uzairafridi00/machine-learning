import os
import time
import cProfile
import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

PROFILES_DIR = 'profiles'
os.makedirs(PROFILES_DIR, exist_ok=True)

app = FastAPI()


@app.middleware('http')
async def create_profile(request: Request, call_next):
    time_stamp = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
    path = request.url.path.strip('/').replace('/', '_') or 'root'
    profile_name = os.path.join(PROFILES_DIR, f'{path}_{time_stamp}.prof')

    profiler = cProfile.Profile()
    profiler.enable()

    response = await call_next(request)

    profiler.disable()
    profiler.dump_stats(profile_name)

    print(f'Profile saved: {profile_name}')
    return response


@app.get('/')
def home():
    return {'message': 'cProfile demo'}


@app.get('/compute')
async def compute():
    time.sleep(1)
    result = sum((i * 2) for i in range(10000))
    return JSONResponse({'result': result})