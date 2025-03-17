import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from typing import Literal, Optional
from uuid import uuid5, NAMESPACE_DNS
import random
from pydantic import BaseModel

class Book(BaseModel):
    name: str
    genre: Literal['fiction', 'non-fiction']
    price: float
    book_id: Optional[str] = None

BOOKS_FILE = "books.json"
BOOKS = []

if os.path.exists(BOOKS_FILE):
    with open(BOOKS_FILE, "r") as f:
        BOOKS = json.load(f)

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Welcome to the Book Store!"}

@app.get("/random-book")
async def random_book():
    return random.choice(BOOKS)

@app.get("/list-books")
async def list_books():
    return {"books": BOOKS}

@app.get("/get-book/{book_id}")
async def get_book_by_index(index: int):
    try:
        return BOOKS[index]
    except IndexError:
        raise HTTPException(status_code=404, detail="Book not found")
    
@app.post("/add-book")
async def add_book(book: Book):
    book.book_id = uuid5(NAMESPACE_DNS, book.name).hex
    json_book = jsonable_encoder(book)
    BOOKS.append(json_book)

    with open(BOOKS_FILE, "w") as f:
        json.dump(BOOKS, f, indent=4)
    return {"book_id": book.book_id}

@app.delete("/delete-book/{book_id}")
async def delete_book(book_id: str):
    global BOOKS
    BOOKS = [book for book in BOOKS if book["book_id"] != book_id]
    with open(BOOKS_FILE, "w") as f:
        json.dump(BOOKS, f, indent=4)
    return {"message": "Book deleted successfully"}