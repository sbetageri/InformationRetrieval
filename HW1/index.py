import re
import os
import collections
import time
import string


class index:
    def __init__(self, path):
        self.path = path
        self.index = {}
        self.buildIndex()

    def buildIndex(self):
        '''Builds an index of text files in a given directory.
        '''
        # function to read documents from collection, tokenize and build the index with tokens
        # index should also contain positional information of the terms in the document --- term: [(ID1,[pos1,pos2,..]), (ID2, [pos1,pos2,…]),….]
        # use unique document IDs
        files = os.scandir(self.path)
        count = 0
        for doc_file in files:
            docID = doc_file.name
            doc_index = self.build_inverted_index_for_doc(doc_file)
            self._add_doc_index_to_dir_index(docID, doc_index)
            count =+ 1
            if count == 1:
                break
        print(self.index)
    
    def _add_doc_index_to_dir_index(self, docID, doc_index):
        '''Add the index of a document to the index of all the documents.
        
        :param docID: Unique ID for a given document.
        :type docID: String
        :param doc_index: Index of tokens in a document.
        :type doc_index: Dict
        '''
        for token in doc_index:
            if token in self.index:
                self.index[token].append(
                    (docID, doc_index[token])
                )
            else:
                self.index[token] = [
                    (docID, doc_index[token])
                ]
    
    def build_inverted_index_for_doc(self, doc):
        '''Build an index for a given file.
        Structure of dict is 
        {
            'token_1' : [position_1, position_2, ...],
            'token_2' : [position_1, position_2, ...]
        }
        
        :param doc: Absolute path of a given file.
        :type doc: String
        :return: Index of given file.
        :rtype: Dict
        '''
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
                word = self._tokenize(word)
                if word == '':
                    continue
                elif word in index:
                    index[word].append(count)
                else:
                    index[word] = [count]
        return index
    
    def _tokenize(self, word):
        '''Tokenize a given word by removing punctuation and converting to lowercase. 
        
        :param word: String to be tokenized.
        :type word: String
        :return: Tokenized string.
        :rtype: String
        '''
        word = word.translate(str.maketrans('','', string.punctuation))
        word = word.lower()
        return word

    def and_query(self, query_terms):
        # function for identifying relevant docs using the index
        pass

    def print_dict(self):
        # function to print the terms and posting list in the index
        pass

    def print_doc_list(self):
        # function to print the documents and their document id
        pass

if __name__ == '__main__':
    indexer = index('./collection/')