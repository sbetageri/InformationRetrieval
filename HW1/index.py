import re
import os
import collections
import time
import string


class index:
    def __init__(self, path):
        self.path = path
        self.buildIndex()

    def buildIndex(self):
        # function to read documents from collection, tokenize and build the index with tokens
        # index should also contain positional information of the terms in the document --- term: [(ID1,[pos1,pos2,..]), (ID2, [pos1,pos2,…]),….]
        # use unique document IDs
        files = os.scandir(self.path)
        dir_index = {}
        count = 0
        for doc_file in files:
            doc_index = self.build_inverted_index_for_doc(doc_file)
            print(doc_index)
            count =+ 1
            if count == 1:
                break
    
    def build_inverted_index_for_doc(self, doc):
        ## doc must be the name of the DirEntry
        index = {}
        doc_file = open(doc, 'r')
        content = doc_file.read()
        lines = content.split("\n")
        count = 0
        for line in lines:
            ## Empty line
            if len(line) == 0:
                continue
            words = line.split(' ')
            for word in words:
                count += 1
                word = word.translate(str.maketrans('','', string.punctuation))
                if word == '':
                    continue
                elif word in index:
                    print(word)
                    index[word].append(count)
                else:
                    index[word] = [count]
        return index

    def and_query(self, query_terms):
        # function for identifying relevant docs using the index
        pass

    def print_dict(self):
        # function to print the terms and posting list in the index
        pass

    def print_doc_list(self):
        # function to print the documents and their document id
        pass