import json

# utility functions
def load_data():
	with open("patients.json", "r") as f:
		data = json.load(f)
	return data

def save_data(data):
	with open("patients.json", "w") as f:
		json.dump(data, f)