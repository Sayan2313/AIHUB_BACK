# def index(id_num:int):
#     if id_num == 0:
#     response = json.dumps({"message": get_llm`(id_num).response()})
#     return Response(content=response, headers={'Content-Type': 'application/json'})
from fastapi import APIRouter

from aihub_back.core.exceptions import ModelNotFound
from aihub_back.models.LLM.model_instances import get_llm
from aihub_back.schemas.llm import ChatRequestSchema
from aihub_back.schemas.model_ids import ModelIds

llm_router = APIRouter(prefix="/api/model/llm", tags=["LLM"])


@llm_router.post("/")
def response(data: ChatRequestSchema):
    try:
        model_id = ModelIds[data.model_name].value
    except KeyError:
        raise ModelNotFound
    llm = get_llm(model_id)
    return llm.response(data.prompt)
