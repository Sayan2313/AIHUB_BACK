from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from aihub_back.models.LLM.core.rag import RAGManager


class LLMBaseclass(ABC):
    def __init__(self)->None:
        self.rag_manager = RAGManager()
        pdf_path = Path(__file__).resolve().parent.parent / "pdfs" / "india_guide.pdf"
        self.rag_manager.ingest_pdf(str(pdf_path))
    @abstractmethod
    async def response(self,text:str,file: Optional[UploadFile] = None)->str:
        raise NotImplementedError