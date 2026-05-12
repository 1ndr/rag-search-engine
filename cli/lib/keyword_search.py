import os 
import pickle
import string
from collections import defaultdict, Counter

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
        self.term_frequencies:dict[int, Counter] = defaultdict(Counter)
 
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenised_doc = tokenise_text(text)

        for token in set(tokenised_doc):
            self.index[token].add(doc_id)

        for token in tokenised_doc:
            self.term_frequencies[doc_id][token]+= 1

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
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokenised_term = tokenise_text(term)

        if len(tokenised_term) > 1:
            raise ValueError(f"more than one token detected for {term}")

        count = self.term_frequencies[doc_id][tokenised_term[0]]
        if count:
            return count
        return 0


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenise_text(query)
    seen, results = set(), []

    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results
    return results

def tf_command(doc_id: int, token: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, token)


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
