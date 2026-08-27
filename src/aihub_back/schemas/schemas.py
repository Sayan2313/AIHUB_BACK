from typing import Literal

from pydantic import BaseModel, Field


class ModelSchema(BaseModel):
    name: str = Field(...,min_length=1)
    summary: str = Field(...,min_length=1)
    description: str = Field(...,min_length=1)
    capabilities: list[str] = Field(...,min_length=1)
    latency: Literal["Low","Medium","High"] = Field(...)
    price:str = Field(...,min_length=1)

class ModelsInfoSchema(BaseModel):
    models : list[ModelSchema] = Field(...,min_length=1)

class ChatRequestSchema(BaseModel):
    model_name: str
    prompt : str

class ChatResponseSchema(BaseModel):
    model_name: str
    prompt: str
    response: str
