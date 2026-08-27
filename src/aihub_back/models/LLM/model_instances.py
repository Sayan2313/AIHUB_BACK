import httpx

from aihub_back.core.exceptions import ModelUnavailable
from aihub_back.models.LLM.phi3.inference import OllamaPhi3
from aihub_back.models.model_ids import ModelIds


# Instances
def get_llm(id_num:int):
    try:
        if id_num == ModelIds.phi3.value:
            return OllamaPhi3()
    except httpx.ConnectError:
        raise ModelUnavailable()


# chat_history = InMemoryChatMessageHistory()
# current_pdf_index = None
# current_pdf_chunks = []
