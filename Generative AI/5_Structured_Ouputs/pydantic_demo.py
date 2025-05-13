from pydantic import BaseModel, Field # EmailStr
from typing import Optional

class Student(BaseModel):
    name: str = 'John Doe'
    age: Optional[int] = None
    #email: EmailStr
    cgpa: float = Field(gt=0, lt=4, default=2.0, description="CGPA must be between 0 and 4")

new_student = {'age': 32, 'email': 'abc@gmail.com'}
student = Student(**new_student)
student_dict = student.model_dump()
print(student_dict)
student_json = student.model_dump_json()
print(student_json)