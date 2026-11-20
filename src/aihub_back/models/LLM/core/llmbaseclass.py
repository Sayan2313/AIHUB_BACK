from abc import ABC, abstractmethod
from pathlib import Path

from fastapi import UploadFile

from aihub_back.models.LLM.core.rag import RAGManager


class LLMBaseclass(ABC):
    def __init__(self)->None:
        self.rag_manager = RAGManager()
        pdf_path = Path(__file__).resolve().parent.parent / "pdfs" / "basic_conversation_corpus_for_local_lm.pdf"
        if pdf_path.is_file():
            self.rag_manager.ingest_pdf(str(pdf_path))
    @abstractmethod
    async def response(self,text:str,file: UploadFile | None = None)->str:
        raise NotImplementedError