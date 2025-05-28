from pydantic import BaseModel, EmailStr, AnyUrl, Field
from typing import List, Dict, Optional, Annotated

class Patient(BaseModel):
	name: Annotated[str, Field(max_length=50, title="Name of Patient", description="Give the name of patient in less than 50 chars", examples=["Ahmad", "Shahab"])]
	age: Annotated[int, Field(gt=0, lt=120, title="Age of Patient", description="Give the age of patient between 0 to 120", examples=[20, 40])]
	email: EmailStr
	linkedin_url: AnyUrl
	weight: Annotated[float, Field(gt=0, strict=True, title="Weight of Patient", description="Give the weight of patient", examples=[49.5, 60])]
	married: Annotated[bool, Field(default=False, description="Is the patient married or not?")]  # default value will be False
	allergies: Annotated[Optional[List[str]], Field(default=None, max_length=5, description="List down only upto 5 allergies")] # setting the optional field
	contact_details: Dict[str, str]

def insert_patient_data(patient: Patient):
	print(patient.name)
	print(patient.age)
	print(patient.email)
	print(patient.linkedin_url)
	print(patient.weight)
	print(patient.married)
	print(patient.allergies)
	print(patient.contact_details)
	print("Inserted")

patient_info = {
	'name': 'Khan',
	'age': 12,
	'email': 'abc@gmail.com',
	'linkedin_url': 'http://linkedin.com',
	'weight': 2, 
	#'married': False, 
	#'allergies': ['Pollen', 'Dust'], 
	'contact_details': 
		{'phone': '03243284123'}
	}

patient1 = Patient(**patient_info)

insert_patient_data(patient1)