from fastapi import FastAPI, requests

app = FastAPI()

# GET, POST, PUT (update), DELETE
@app.get("/")
def index():
    return {"message": "Hello Python!"}

@app.get("/calculation")
def calculation():
    # do some calculation
    pass
    return "Calculation done"

# asyn function
@app.get("/calculation")
async def get_data_from_db():
    await requests.get("http://localhost:8000/calculation")
    return {"data": "data"}