from fastapi import FastAPI

app = FastAPI()

# GET, POST, PUT (update), DELETE
@app.get("/")
def index():
    return {"message": "Hello Python!"}