from fastapi import FastAPI, Path, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal, Optional
import json

# utilities methods
from utilities import load_data, save_data

# main app
app = FastAPI()

# patient class
class Patient(BaseModel):
	id: Annotated[str, Field(..., description="ID of the Patient", example="P001")]
	name: Annotated[str, Field(..., description="Name of the Patient", example="Khan")]
	city: Annotated[str, Field(..., description="Name of the City", example="Peshawar")]
	age: Annotated[int, Field(..., gt=0, lt=120, description="Age of the Patient", example="20")]
	gender: Annotated[Literal["male", "female", "others"], Field(..., description="Gender of the Patient", example="male or female")]
	height: Annotated[float, Field(..., gt=0, description="Height of the Patient in meters", example="5.6")]
	weight: Annotated[float, Field(..., gt=0, description="Weight of the Patient in Kg", example="60.3")]

	@computed_field
	@property
	def bmi(self) -> float:
		bmi = round(self.weight / (self.height ** 2), 2)
		return bmi

	@computed_field
	@property
	def verdict(self) -> str:
		if self.bmi < 18.5:
			return "Underweight"
		elif self.bmi < 25:
			return "Normal"
		elif self.bmi < 30:
			return "Overweight"
		else:
			return "Obese"


# patient class for Update
class PatientUpdate(BaseModel):
	name: Annotated[Optional[str], Field(default=None)]
	city: Annotated[Optional[str], Field(default=None)]
	age: Annotated[Optional[int], Field(default=None, gt=0)]
	gender: Annotated[Optional[Literal['male', 'female', 'others']], Field(default=None)]
	height: Annotated[Optional[float], Field(default=None, gt=0)]
	weight: Annotated[Optional[float], Field(default=None, gt=0)]


@app.get("/")
def home():
	return {"message": "Patient Management System"}

@app.get("/about")
def about():
	return {"message": "A fully functional API to manage your patient records"}

@app.get("/view")
def view():
	data = load_data()
	return data

@app.get("/patient/{patient_id}")
def view_patient(patient_id: str = Path(..., description="ID of the Patient in the DB", example="P001")):

	# load the data
	data = load_data()

	if patient_id in data:
		return data[patient_id]

	raise HTTPException(status_code=404, detail="Patient record not found.")

@app.get("/sort")
def sort_patients(sort_by: str = Query(..., description="Sort on the basis of height, weight or bmi"), order: str = Query("asc", description="sort in asc or desc order")):

	valid_fields = ["height", "weight", "bmi"]

	if sort_by not in valid_fields:
		raise HTTPException(status_code=400, detail=f"Invalid field select from {valid_fields}")

	if order not in ['asc', 'desc']:
		raise HTTPException(status_code=400, detail=f"Invalid order select. Please select asc or desc")

	data = load_data()

	sort_order = True if order=='desc' else False

	sorted_data = sorted(data.values(), key=lambda x: x.get(sort_by, 0), reverse=sort_order)

	return sorted_data 


@app.post("/create")
def create_patient(patient: Patient):

	# load existing data
	data = load_data()

	# check if patient already exist
	if patient.id in data:
		raise HTTPException(status_code=400, detail="Patient already exists")

	# new patient add to the database
	data[patient.id] = patient.model_dump(exclude=["id"])

	# save into the json
	save_data(data)

	return JSONResponse(status_code=201, content={"message": "Patient created successfully"})


@app.put('/edit/{patient_id}')
def update_patient(patient_id: str, patient_update: PatientUpdate):

    data = load_data()

    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    
    existing_patient_info = data[patient_id]

    updated_patient_info = patient_update.model_dump(exclude_unset=True)

    for key, value in updated_patient_info.items():
        existing_patient_info[key] = value

    #existing_patient_info -> pydantic object -> updated bmi + verdict
    existing_patient_info['id'] = patient_id
    patient_pydandic_obj = Patient(**existing_patient_info)
    #-> pydantic object -> dict
    existing_patient_info = patient_pydandic_obj.model_dump(exclude='id')

    # add this dict to data
    data[patient_id] = existing_patient_info

    # save data
    save_data(data)

    return JSONResponse(status_code=200, content={'message':'Patient updated successfully'})


@app.delete("/delete/{patient_id}")
def delete_patient(patient_id: str):
	# load the data
 	data = load_data()

 	# check for data present
 	if patient_id not in data:
 		raise HTTPException(status_code=404, detail="Patient not found")

 	# delete the patient
 	del data[patient_id]

 	# save data
 	save_data(data)

 	return JSONResponse(status_code=200, content={"message": "Patient successfully deleted"})
