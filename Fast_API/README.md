# FastAPI Notes

## What is FastAPI?
FastAPI is a modern, high-performance, web framework for building APIs with Python 3.7+ based on standard Python type hints. It is built on Starlette for web parts and Pydantic for data validation.

### Key Features:
- **Fast**: As the name suggests, it is one of the fastest Python frameworks.
- **Easy to Use**: Simple syntax and automatic API documentation generation.
- **Data Validation**: Uses Pydantic for request validation.
- **Asynchronous Support**: Supports both synchronous and asynchronous programming.
- **Automatic Interactive Documentation**: OpenAPI and Swagger UI are built-in.

## Why FastAPI?
- **Performance**: Almost on par with Node.js and Go.
- **Type Safety**: Leverages Python type hints for automatic validation.
- **Built-in Documentation**: Provides interactive API docs with Swagger UI and ReDoc.
- **Asynchronous Support**: Native support for async/await for high concurrency.
- **Easy Deployment**: Works with Uvicorn, Gunicorn, or any ASGI server.

## What is an API?
An API (Application Programming Interface) allows different applications or services to communicate with each other.
In FastAPI, an API is created using Python functions mapped to HTTP methods like `GET`, `POST`, `PUT`, and `DELETE`.

### Example:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
```

## Asynchronous in FastAPI
FastAPI fully supports asynchronous programming using `async def`. This allows handling multiple requests concurrently without blocking operations.

### Example:
```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get("/async")
async def async_endpoint():
    await asyncio.sleep(2)
    return {"message": "This was an async response!"}
```

## Additional Notes

### Request Body and Data Validation
FastAPI uses Pydantic models to validate and structure request payloads.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    in_stock: bool

@app.post("/items/")
def create_item(item: Item):
    return {"item_name": item.name, "item_price": item.price}
```

### Path and Query Parameters
```python
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}
```

### Dependency Injection
FastAPI has built-in support for dependency injection, making code modular and testable.
```python
from fastapi import Depends

def get_db():
    return {"db": "connected"}

@app.get("/db")
def read_db(db=Depends(get_db)):
    return db
```

### Middleware in FastAPI
Middleware allows processing requests before they reach endpoints.
```python
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Technical Terms in FastAPI
- **ASGI (Asynchronous Server Gateway Interface)**: The standard interface between async-capable Python web servers and applications.
- **Pydantic**: A data validation library used in FastAPI for request and response validation.
- **Starlette**: A lightweight ASGI framework that FastAPI is built upon.
- **Dependency Injection**: A design pattern used to provide dependencies to a function or class dynamically.
- **Middleware**: A function that runs before and/or after request handling in FastAPI to modify requests or responses.
- **Background Tasks**: Used for running tasks in the background without blocking the main request/response cycle.
- **CORS (Cross-Origin Resource Sharing)**: A mechanism that allows or restricts resources from being accessed by different origins.
- **OpenAPI**: A specification for describing APIs that allows automatic generation of API documentation.
- **Uvicorn**: An ASGI server for running FastAPI applications in production.

### Running FastAPI Server
To start the FastAPI server, use Uvicorn:
```sh
uvicorn main:app --reload

fastapi dev main.py --port 9999
```

This command starts the server with auto-reload enabled, useful for development.

## FastAPI vs Flask

Choosing between **FastAPI** and **Flask** depends on your project requirements. Here's a comparison to help you decide:

### **When to Use FastAPI**
Use **FastAPI** if you need:
1. **High Performance** – It is much faster than Flask due to ASGI and async support.
2. **Asynchronous Processing** – If you have I/O-bound operations like database queries, external API calls, or WebSockets.
3. **Type Safety** – If you want built-in request validation and type hinting (reduces bugs).
4. **Automatic Documentation** – FastAPI provides Swagger UI and ReDoc out of the box.
5. **Modern Python Features** – If you prefer Python 3.7+ features like `dataclasses`, `async/await`, and `type hints`.
6. **Production APIs** – For large-scale, high-performance applications (e.g., ML APIs, microservices).

### **When to Use Flask**
Use **Flask** if you need:
1. **Simplicity** – Flask is lightweight, easier to learn, and has minimal boilerplate.
2. **Synchronous Processing** – If your API doesn’t require async capabilities.
3. **More Flexibility** – Flask allows choosing your preferred tools for validation, authentication, and database ORM.
4. **Large Ecosystem** – Flask has more third-party extensions due to its long history.
5. **Quick Prototyping** – If you want to build a simple web application or API fast with minimal setup.

### **Summary: Which One Should You Choose?**
| Feature      | FastAPI  | Flask  |
|-------------|---------|--------|
| Performance | 🚀 High (async) | 🐢 Slower (sync) |
| Async Support | ✅ Yes | ❌ No (without third-party libraries) |
| Type Safety | ✅ Yes (Pydantic) | ❌ No |
| Built-in Docs | ✅ Yes (Swagger, ReDoc) | ❌ No (Needs Flask-RESTPlus) |
| Learning Curve | 📈 Medium | 📉 Easy |
| Best for | APIs, ML Services, Async Apps | Prototyping, Small Apps |

### **Final Decision**
- If you need **speed, async, and modern Python**, choose **FastAPI**.
- If you need **simplicity, flexibility, and a lightweight framework**, choose **Flask**.