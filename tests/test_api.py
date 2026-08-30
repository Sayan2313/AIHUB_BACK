from fastapi.testclient import TestClient
from aihub_back.main import app
from aihub_back.schemas.schemas import ChatRequestSchema

client = TestClient(app)

def test_model_info_check():
    res = client.get(url = "/api/models/info")
    assert res.status_code == 200
def test_llm_model():
    payload : ChatRequestSchema =  ChatRequestSchema(
        model_name="phi3",
        prompt="What are you?",
    )
    res = client.post(url = "/api/models/llm",json=payload.model_dump())
    assert res.status_code == 200