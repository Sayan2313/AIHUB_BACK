from typing import Annotated, Optional

from fastapi import APIRouter, UploadFile, File, Form

from aihub_back.core.exceptions import ModelNotFound
from aihub_back.models.LLM.model_instances import get_llm
from aihub_back.models.model_ids import ModelIds
from aihub_back.schemas.schemas import ChatResponseSchema

llm_router = APIRouter(prefix="/api/models", tags=["LLM"])


@llm_router.post("/llm",response_model=ChatResponseSchema)
def response(
    model_name: Annotated[str,Form(...)],
    prompt : Annotated[str,Form(...)],
    file : Annotated[Optional[UploadFile | None], File()] = None):
    try:
        model_id = ModelIds[model_name].value
    except KeyError:
        raise ModelNotFound
    llm = get_llm(model_id)
    res : str = llm.response(prompt)
    return {'model_name': model_name, 'prompt': prompt, 'response' : res}
