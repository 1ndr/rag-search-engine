import pickle
import os
from .keyword_search import tokenise_text
from .search_utils import load_movies
 
class InvertedIndex:
    def __init__(self, index, docmap):
        self.index = index
        self.docmap = docmap

    def __add_document(self, doc_id, text):
        tokenised_document = tokenise_text(text)

        for token in tokenised_document:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}

    def get_documents(self, term):
        return sorted(self.index[term])

    def build(self):
        movies = load_movies()

        for movie in movies:
            self.index.__add_document(movie['id'], f"{movie['title']}{movie['description']}")
            self.docmap[movie['id']] = movie

    def save():



        
        
        
        
