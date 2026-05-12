import os 
import pickle
import string
from collections import defaultdict

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
)

from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap:dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenised_doc = tokenise_text(text)

        for token in set(tokenised_doc):
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def build(self) -> None:
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            doc_desc = f"{movie['title']} {movie['description']}"

            self.__add_document(doc_id, doc_desc)
            self.docmap[doc_id] = movie

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    tokenise_query = tokenise_text(query)

    try:
        movies_index = InvertedIndex()
        movies_index.load()
    except FileNotFoundError:
        print("File not found in cache")
        return

    results = []

    return results


def has_mathing_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def tokenise_text(text: str) -> list[str]:
    stopwords = load_stopwords()
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []

    for token in tokens:
        if token and token not in stopwords:
            valid_tokens.append(token)

    stemmed_tokens = []
    stemmer = PorterStemmer()
    for token in valid_tokens:
        stemmed_tokens.append(stemmer.stem(token))

    return stemmed_tokens
