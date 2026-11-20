from io import BytesIO

from fastapi import UploadFile
from langchain_ollama import OllamaLLM
from pypdf import PdfReader

from aihub_back.models.LLM.core.llmbaseclass import LLMBaseclass


class OllamaPhi3(LLMBaseclass):
    def __init__(self):
        super().__init__()
        self.model = OllamaLLM(model="phi3:mini",num_ctx=8192,num_predict=1024,temperature=0.7,top_p=0.9)
    async def response(self,text:str,file: UploadFile | None = None)->str:
        # 1. Process an uploaded PDF, if provided
        if file is not None:
            contents = await file.read()

            reader = PdfReader(BytesIO(contents))
            document_text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

            if not document_text.strip():
                raise ValueError(
                    "The uploaded PDF contains no extractable text."
                )
            self.rag_manager.ingest_text(
                document_text,
                source=file.filename or "uploaded.pdf",
            )

        # 2. Retrieve the top 3 relevant chunks
        text_chunks = self.rag_manager.retrieve(
            question=text,
            top_k=3,
        )
        context = "\n\n".join(text_chunks)
        enhanced_prompt = (f"""<|system|>
        You are a conversational helpful assistant.
        Answer in 1-3 sentences using simple, direct language.
        If context is provided, use it to answer. If the answer is missing,
        say you don't know based on the provided context.
        <|end|>
        <|user|>
        Context:
        {context}
    
        Question:
        {text}
        <|end|>
        <|assistant|>
        """).strip()

        res = await self.model.ainvoke(enhanced_prompt)

        return res.strip()
