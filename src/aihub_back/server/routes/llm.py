from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from aihub_back.models.LLM.model_instances import get_llm
from aihub_back.models.model_ids import ModelIds
from aihub_back.server.core.exceptions import ModelNotFound
from aihub_back.server.schemas.schemas import ChatResponseSchema

llm_router = APIRouter(prefix="/api/models", tags=["LLM"])


@llm_router.post("/llm",response_model=ChatResponseSchema)
async def response(
    model_name: Annotated[str,Form(...)],
    prompt : Annotated[str,Form(...)],
    file : Annotated[UploadFile | None, File()] = None):
    try:
        model_id = ModelIds[model_name].value
    except KeyError:
        raise ModelNotFound
    llm = get_llm(model_id)
    res : str = await llm.response(prompt,file)
    return {'model_name': model_name, 'prompt': prompt, 'response' : res}
