from pydantic import BaseModel


class ChatRequestSchema(BaseModel):
    model_name: str
    prompt : str
