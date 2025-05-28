from pydantic import BaseModel

class Address(BaseModel):
    city: str
    country: str
    zipcode: int

class Patient(BaseModel):
    name: str
    gender: str
    age: int
    address: Address

address_dict = {'city': 'Peshawar', 'country': 'Pakistan', 'zipcode': 25000}
address1 = Address(**address_dict)

patient_dict = {'name': 'Ahmad', 'gender': 'Male', 'age': 23, 'address': address1}
patient1 = Patient(**patient_dict)
print(patient1)

temp = patient1.model_dump()
temp2 = patient1.model_dump_json()
print(temp2)
print(type(temp2))