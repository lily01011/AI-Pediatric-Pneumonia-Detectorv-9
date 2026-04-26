import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest


class DocumentProcessor:
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        self.reranker = Ranker()

    def process_file(self, file):
        """Parses, chunks, and indexes a PDF file."""
        text = ""
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        else:
            text = file.read().decode("utf-8")

        chunks = self.text_splitter.split_text(text)

        # Vector index
        embeddings = self.embed_model.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))

        # BM25 index
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        return {"chunks": chunks, "index": index, "bm25": bm25}

    def hybrid_search_and_rerank(self, query, kb, top_k=3):  # top_k added
        """Combines Vector + BM25 and applies Reranking."""
        query_vec = self.embed_model.encode([query])
        _, v_indices = kb["index"].search(np.array(query_vec).astype('float32'), top_k * 2)

        tokenized_query = query.lower().split()
        bm25_scores = kb["bm25"].get_top_n(tokenized_query, kb["chunks"], n=top_k * 2)

        candidate_chunks = list(set(
            [kb["chunks"][i] for i in v_indices[0]] + bm25_scores
        ))

        passages = [{"id": i, "text": c} for i, c in enumerate(candidate_chunks)]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.reranker.rerank(rerank_request)

        return [r['text'] for r in results[:top_k]]