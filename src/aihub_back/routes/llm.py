# def index(id_num:int):
#     if id_num == 0:
#     response = json.dumps({"message": get_llm`(id_num).response()})
#     return Response(content=response, headers={'Content-Type': 'application/json'})
from fastapi import APIRouter

from aihub_back.core.exceptions import ModelNotFound
from aihub_back.models.LLM.model_instances import get_llm
from aihub_back.models.model_ids import ModelIds
from aihub_back.schemas.schemas import ChatRequestSchema, ChatResponseSchema

llm_router = APIRouter(prefix="/api/model/llm", tags=["LLM"])


@llm_router.post("",response_model=ChatResponseSchema)
def response(data: ChatRequestSchema):
    try:
        model_id = ModelIds[data.model_name].value
    except KeyError:
        raise ModelNotFound
    llm = get_llm(model_id)
    res : str = llm.response(data.prompt)
    return {'model_name': data.model_name, 'prompt': data.prompt, 'response' : res}
