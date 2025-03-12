from typing import Optional, List
from enum import IntEnum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

class Priority(IntEnum):
    LOW = 3
    MEDIUM = 2
    HIGH = 1

class TodoBase(BaseModel):
    todo_name: str = Field(..., min_length=3, max_length=512, description="Name of the Todo")
    todo_description: str = Field(description="Description of the Todo")
    priority: Priority = Field(default = Priority.LOW, description="Priority of the Todo")

class TodoCreate(TodoBase):
    pass

class Todo(TodoBase):
    todo_id: int = Field(..., description="Unique Identifier of the Todo")

class TodoUpdate(TodoBase):
    todo_name: Optional[str] = Field(default = None, min_length=3, max_length=512, description="Name of the Todo")
    todo_description: Optional[str] = Field(default = None, description="Description of the Todo")
    priority: Optional[Priority] = Field(default = None, description="Priority of the Todo")


all_todos = [
    Todo(todo_id=1, todo_name="Learn Python", todo_description="Learn Python Programming Language", priority=Priority.HIGH),
    Todo(todo_id=2, todo_name="Learn FastAPI", todo_description="Learn FastAPI Framework", priority=Priority.MEDIUM),
    Todo(todo_id=3, todo_name="Learn Django", todo_description="Learn Django Framework", priority=Priority.LOW),
    Todo(todo_id=4, todo_name="Learn Flask", todo_description="Learn Flask Framework", priority=Priority.HIGH),
    Todo(todo_id=5, todo_name="Learn JavaScript", todo_description="Learn JavaScript Programming Language", priority=Priority.MEDIUM),
    # {"todo_id": 1, "todo_name": "Learn Python", "todo_description": "Learn Python Programming Language"},
    # {"todo_id": 2, "todo_name": "Learn FastAPI", "todo_description": "Learn FastAPI Framework"},
    # {"todo_id": 3, "todo_name": "Learn Django", "todo_description": "Learn Django Framework"},
    # {"todo_id": 4, "todo_name": "Learn Flask", "todo_description": "Learn Flask Framework"},
    # {"todo_id": 5, "todo_name": "Learn JavaScript", "todo_description": "Learn JavaScript Programming Language"},
]

# @app.get("/")
# def home():
#     return {"message": "Welcome to TODO App"}

@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int):
    for todo in all_todos:
        if todo.todo_id == todo_id:
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")

# query parameter
@app.get("/todos", response_model=List[Todo])
def get_todos(first_n: int = None):
    if first_n:
        return {"todos": all_todos[:first_n]}
    else:
        return {"todos": all_todos}  
    
# POST request
@app.post("/todos", response_model=Todo)
def create_todo(todo: TodoCreate):
    new_todo_id = max(todo.todo_id for todo in all_todos) + 1

    new_todo = Todo(
        todo_id = new_todo_id,
        todo_name = todo.todo_name,
        todo_description = todo.todo_description,
        priority = todo.priority
    )

    all_todos.append(new_todo)
    return new_todo


# PUT request
@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, updated_todo: TodoUpdate):
    for todo in all_todos:
        if todo.todo_id == todo_id:
            if todo.todo_name is not None:
                todo.todo_name = updated_todo.todo_name
            if todo.todo_description is not None:
                todo.todo_description = updated_todo.todo_description,
            if todo.todo_id is not None:
                todo.priority = updated_todo.priority
            return {"message": "Todo updated successfully"}
    raise HTTPException(status_code=404, detail="Todo not found")


# DELETE request
@app.delete("/todos/{todo_id}", response_model=Todo)
def delete_todo(todo_id: int):
    for index, todo in enumerate(all_todos):
        if todo.todo_id == todo_id:
            deleted_todo = all_todos.pop(index)
            return deleted_todo
    raise HTTPException(status_code=404, detail="Todo not found")

