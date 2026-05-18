import os
import numpy as np

from .search_utils import (
    CACHE_DIR,
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

    def load_or_create_embeddings(self, documents: list[dict]) -> list[float]:
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

    def generate_embedding(self, text: str) -> list[float]:
        if text == '' or not text:
            raise ValueError('text must not be empty')
        return self.model.encode([text])[0]


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
