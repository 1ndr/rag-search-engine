Class InvertedIndex:
    def __init__(self, index, docmap):
        self.index = index
        self.docmap = docmap

    def __add_document(self, doc_id, text):
        
