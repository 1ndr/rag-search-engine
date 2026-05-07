from .keyword_search import tokenise_text

Class InvertedIndex:
    def __init__(self, index, docmap):
        self.index = index
        self.docmap = docmap

    def __add_document(self, doc_id, text):
        tokenise_text = tokenise_text(text)
        
        
