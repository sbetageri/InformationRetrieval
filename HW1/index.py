import re
import os
import collections
import time
import string


class index:
    def __init__(self, path):
        '''Constructor
        
        :param path: Path to directory to index.
        :type path: String
        '''
        self.path = path
        self.index = {}
        self.token_freq = {}
        self.doc_list = {}
        self.buildIndex()

    def buildIndex(self):
        '''Builds an index of text files in a given directory.
        '''
        # function to read documents from collection, tokenize and build the index with tokens
        # index should also contain positional information of the terms in the document --- term: [(ID1,[pos1,pos2,..]), (ID2, [pos1,pos2,…]),….]
        # use unique document IDs
        print('Building index')
        files = os.scandir(self.path)
        count = 0
        for doc_file in files:
            docID = doc_file.name

            if docID not in self.doc_list:
                self.doc_list[docID] = (doc_file.path, doc_file.name)

            print('Indexing : ', docID)
            doc_index = self.build_inverted_index_for_doc(doc_file)
            self._add_doc_index_to_dir_index(docID, doc_index)
        print('Index built')
    
    def _add_doc_index_to_dir_index(self, docID, doc_index):
        '''Add the index of a document to the index of all the documents.
        Also build a frequency counter for the terms

        Structure of index is 
        {
            'token_1' : [
                (docID_a, [position_1, position_2, ...]),
                (docID_b, [position_1, position_2, ...]),
                (docID_c, [position_1, position_2, ...])
            ],
            'token_2' : [
                (docID_p, [position_1, position_2, ...]),
                (docID_q, [position_1, position_2, ...]),
                (docID_r, [position_1, position_2, ...])
            ]
            .
            .
            .
        }

        Structure of term_freq is 
        {
            'token_1' : count_1,
            'token_2' : count_2,
            .
            .
            .
        }
        
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
            if token in self.token_freq:
                self.token_freq[token] += 1
            else:
                self.token_freq[token] = 1
    
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

    def and_query(self, query):
        '''Perform a boolean query
        
        :param query: Query entered by user.
        :type query: String
        '''
        # function for identifying relevant docs using the index
        query = self._clean_query(query)
        query_term_freq = self._get_term_freq(query)
        sorted_terms = sorted(query_term_freq, key=lambda freq : freq[0])

        # Avoid MAGIC NUMBERS below.
        doc_set = self._get_docs_for_term(sorted_terms[0][1])
        for term in sorted_terms[1:]:
            term_doc = self._get_docs_for_term(term[1])
            doc_set.intersection(term_doc)
        print('The documents are : ')
        for doc in doc_set:
            print(doc)
    
    def _get_docs_for_term(self, term):
        '''Get a set of document IDs where the term appears.
        
        :param term: Term for which docIDs are needed.
        :type term: String
        :return: Set of docIDs
        :rtype: Set
        '''
        postings = self.index[term]
        docs = set()
        for posting in postings:
            docs.add(posting[0])
        return docs
    
    def _get_term_freq(self, query):
        '''Get the frequency with which a given term appears in the index.
        Returned list has the following structure:
        [
            (term_freq, term_1),
            (term_freq, term_2),
            .
            .
            .
        ]
        
        :param query: Query terms
        :type query: List
        :return: List of terms and their corresponding frequencies in the entire index.
        :rtype: List of tuples.
        '''
        # query_term_freq will be a list with the following structure
        # [
        #   (freq, term)
        # ]
        query_term_freq = []
        term_not_in_index = []
        for term in query:
            if term in self.token_freq:
                freq = self.token_freq[term]
                query_term_freq.append((freq, term))
            else:
                print(term, ' not in index. Try again.')
                assert False
        return query_term_freq
    
    def _clean_query(self, query):
        '''Seperates AND and the remaining terms.
        
        :param query: Query as entered by user.
        :type query: String
        :return: List of query terms.
        :rtype: List of strings
        '''
        query = query.lower()
        query = query.replace('and', '')
        query = query.split()
        return query

    def print_dict(self):
        '''Print index
        '''
        # function to print the terms and posting list in the index
        for token in self.index:
            print('Token : ', token)
            for posting in self.index[token]:
                print('\t', posting[0], ' : ', posting[1])

    def print_doc_list(self):
        '''Print mapping between docID, document name and path to document.
        '''
        # function to print the documents and their document id
        for doc in self.doc_list:
            path, name = self.doc_list[doc]
            print(doc, ' : ', name, ' : ', path)

if __name__ == '__main__':
    start_time = time.time()
    indexer = index('./collection/')
    end_time = time.time()
    time_taken = end_time - start_time
    print('Time Taken to index : ', time_taken)
    query = input('Enter the boolean AND query : ')
    print('The query is : ', query)
    start_time = time.time()
    indexer.and_query(query)
    end_time = time.time()
    time_taken = end_time - start_time
    print('Time taken to retrieve documents : ', time_taken)