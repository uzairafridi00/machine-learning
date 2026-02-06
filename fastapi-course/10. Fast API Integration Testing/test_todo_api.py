import requests

ENDPOINT = "https://todo.pixegami.io"

response = requests.get(ENDPOINT)
print(response)

data = response.json()
print(data)

status_code = response.status_code
print(status_code)

def test_can_call_endpoint():
    response = requests.get(ENDPOINT)
    assert response.status_code == 200

def test_can_create_task():
    payload = {
    "content": "my test content",
    "user_id": "test_user",
    "is_done": False
    }
    response = requests.put(ENDPOINT + "/create-task", json=payload)
    assert response.status_code == 200

    data = response.json()
    print(data)

def test_can_get_tasks():
    