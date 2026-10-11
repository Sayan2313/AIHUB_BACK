from typing import List
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

class RAGManager:
    def __init__(self):

        # Embedding model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # MongoDB: ordinary document storage
        self.client = MongoClient("mongodb://localhost:27017/")
        self.collection = self.client["rag_db"]["documents"]

        # Text chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
        )

    def ingest_text(self, text: str, source: str):
        chunks = self.splitter.split_text(text)

        if not chunks:
            return

        self.collection.delete_many({"source": source})

        embeddings = self.embedder.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        documents = [
            {
                "source": source,
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding.tolist(),
            }
            for i, (chunk, embedding) in enumerate(
                zip(chunks, embeddings)
            )
        ]

        self.collection.insert_many(documents)

    def ingest_pdf(self, pdf_path: str):
        # Extract PDF text
        reader = PdfReader(pdf_path)
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        chunks = self.splitter.split_text(text)

        if not chunks:
            print("No extractable text found.")
            return

        # Remove previously ingested chunks from this source
        self.collection.delete_many({"source": pdf_path})

        # Generate L2-normalized embeddings
        embeddings = self.embedder.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        documents = [
            {
                "source": pdf_path,
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding.tolist(),
            }
            for i, (chunk, embedding) in enumerate(
                zip(chunks, embeddings)
            )
        ]

        self.collection.insert_many(documents)
        print(f"Ingested {len(chunks)} chunks.")

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        # Embed the query using the same model
        query_vector = self.embedder.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Load stored embeddings from MongoDB
        documents = self.collection.find(
            {},
            {"_id": 0, "text": 1, "embedding": 1},
        )

        scored = []

        for doc in documents:
            embedding = np.asarray(
                doc["embedding"], dtype=np.float32
            )

            # Both vectors are normalized, so dot product = cosine similarity
            score = float(np.dot(query_vector, embedding))
            scored.append((score, doc["text"]))

        # Highest similarity first
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            text for score, text in scored[:top_k]
        ]
