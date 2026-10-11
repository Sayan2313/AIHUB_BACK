import httpx

from aihub_back.models.LLM.peak.inference import PeakBase
from aihub_back.models.LLM.phi3.inference import OllamaPhi3
from aihub_back.models.model_ids import ModelIds
from aihub_back.server.core.exceptions import ModelUnavailable


# Instances
def get_llm(id_num:int):
    try:
        if id_num == ModelIds.phi3.value:
            return OllamaPhi3()
        if id_num == ModelIds.peak.value:
            return PeakBase()
    except httpx.ConnectError:
        raise ModelUnavailable()

