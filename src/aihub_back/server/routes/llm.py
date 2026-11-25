from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile

from aihub_back.server.core.exceptions import ModelNotFound
from aihub_back.server.schemas.schemas import ChatResponseSchema

llm_router = APIRouter(prefix="/api/models", tags=["LLM"])


@llm_router.post("/llm",response_model=ChatResponseSchema)
async def response(
    request: Request,
    model_name: Annotated[str,Form(...)],
    prompt : Annotated[str,Form(...)],
    file : Annotated[UploadFile | None, File()] = None,):
    # Find LM
    llm = None
    if model_name == "phi3":
        llm = request.app.state.ollama_llm
    elif model_name == "peak":
        llm = request.app.state.peak_lm
    # Response
    if llm is not None:
        res : str = await llm.response(prompt,file)
        return {'model_name': model_name, 'prompt': prompt, 'response' : res}
    else:
        raise ModelNotFound
