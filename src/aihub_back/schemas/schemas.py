from typing import Literal, Annotated

from pydantic import BaseModel, Field
from fastapi import UploadFile , File


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
    model_name: Annotated[str,Field(min_length=1,max_length=50)]
    prompt : Annotated[str,Field(min_length=1)]
    file : Annotated[UploadFile | None, File()] = None

class ChatResponseSchema(BaseModel):
    model_name: Annotated[str,Field(min_length=1,max_length=50)]
    prompt: Annotated[str,Field(min_length=1)]
    response: Annotated[str,Field(min_length=1)]
