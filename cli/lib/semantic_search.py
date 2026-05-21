import re
import os
import numpy as np

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    load_movies
)
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map:dict[int, dict] = {}

        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def build_embeddings(self, documents: list[dict]) -> list[float]:
        self.__populate_documents(documents)

        list_of_movie_strings = []
        for doc in documents:
            list_of_movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(list_of_movie_strings, show_progress_bar = True)

        os.makedirs(os.path.dirname(self.embe))
        with open(self.embeddings_path, "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> list[list[float]]:
        self.__populate_documents(documents)

        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, "rb") as f:
                self.embeddings = np.load(f)
                if len(self.embeddings) == len(documents):
                    return self.embeddings

        return self.build_embeddings(documents)
    
    def __populate_documents(self, documents: list[dict]) -> None:
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

    def generate_embedding(self, text: str) -> list[list[float]]:
        if text == '' or not text:
            raise ValueError('text must not be empty')
        return self.model.encode([text])[0]

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[str]:
        if self.embeddings is None or len(self.embeddings)==0:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        query_embedding = self.generate_embedding(query)

        scores_lst = []
        for i in range(len(self.embeddings)):
            doc = self.documents[i]
            doc_vec = self.embeddings[i]
            cos_sim = self.cosine_similarity(doc_vec, query_embedding)
            doc["score"] = cos_sim
            scores_lst.append((cos_sim, doc))

        sorted_scores = sorted(scores_lst, key=lambda x: x[0], reverse=True)
        return sorted_scores[:limit]



def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")
    

def embed_text(text: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    search_instance = SemanticSearch()
    movies = load_movies()
    embeddings = search_instance.load_or_create_embeddings(movies)
    
    print(f"Number of docs: {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    

def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)
    return search_instance.search(query, limit)


def chunk_command(text: str, chunk_size: int=DEFAULT_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    return chunking_input(words, chunk_size, overlap)


def semantic_chunk_command(text: str, chunk_size: int=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return chunking_input(sentences, chunk_size, overlap)


def chunking_input(input: list[str], chunk_size: int=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP) -> list[str]:
    return_lst = []
    n_elements = len(input)
    i = 0
    while i < n_elements:
        chunk = input[i:i + chunk_size]
        if return_lst and len(chunk) <= overlap:
            break
        return_lst.append(chunk)
        i+=chunk_size - overlap
    return return_lst
