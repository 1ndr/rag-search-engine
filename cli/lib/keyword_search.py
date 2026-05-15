import math
import os 
import pickle
import string

from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    BM25_TF_COMPONENT_K1_DEFAULT,
    BM25_TF_COMPONENT_B_DEFAULT,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap:dict[int, dict] = {}
        self.term_frequencies:dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths:dict[int, Counter] = defaultdict(Counter)

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenised_doc = tokenise_text(text)

        for token in set(tokenised_doc):
            self.index[token].add(doc_id)

        for token in tokenised_doc:
            self.term_frequencies[doc_id][token]+= 1

        self.doc_lengths[doc_id] = len(tokenised_doc)

    def __get_avg_doc_length(self) -> float:
        if self.doc_lengths:
            count = 0
            for v in self.doc_lengths.values():
                count += v 
            return count / len(self.doc_lengths)
        return 0.0

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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokenised_term = tokenise_text(term)

        if len(tokenised_term) != 1:
            raise ValueError("term must be a single token")

        count = self.term_frequencies[doc_id][tokenised_term[0]]

        if count:
            return count
        return 0

    def get_bm25_tf(self, doc_id: int, term: str, k1: float =BM25_TF_COMPONENT_K1_DEFAULT, b: float = BM25_TF_COMPONENT_B_DEFAULT) -> float:
        raw_tf = self.get_tf(doc_id, term) 
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    def get_idf(self, term:str) -> float:
        tokens = tokenise_text(term)
        if len(tokens) !=1:
            raise ValueError("term must be a single token")

        token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(token))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term:str) -> float:
        tokens = tokenise_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(token))
        return math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)

    def get_tf_idf(self, doc_id:int, term:str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def bm25(self, doc_id:int, term:str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query:str, limit:int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        tokenise_query = tokenise_text(query)
        scores:dict[int, float] = {}
        for doc_id in self.docmap:
            total = 0
            for token in tokenise_query:
                total+= self.bm25(doc_id, token)
            scores[doc_id] = self.docmap[doc_id]
            scores[doc_id]['bm25_score'] = total

        sorted_scores = sorted(scores.items(), key=lambda item: item[1]['bm25_score'], reverse=True)
        return sorted_scores[:limit]



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

def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)


def tf_command(doc_id: int, token: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, token)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: int =BM25_TF_COMPONENT_K1_DEFAULT, b: int = BM25_TF_COMPONENT_B_DEFAULT) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def tf_idf_command(doc_id: int, term:str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)


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
