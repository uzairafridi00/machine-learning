from fastapi import FastAPI

app = FastAPI() # create instance of app

# define route
@app.get("/")
async def home():
    return {"message": "Hello World"}

# add path paramters
@app.get("/items")
async def items():
    return {"message": "This route will list items"}

# add path parameter in the route item as item ids
@app.get("/items/{item_id}")
async def get_items(item_id: int):
    return {"item_id": item_id}

@app.get("/users")
async def get_users():
    return {"message": "This route will return users"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}