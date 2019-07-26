#Python 3.x
import re
import os
import collections
import time
import string
import math
#import other modules as needed

class index:
    def __init__(self,path, stop_words_path):
        self.path = path
        self.index = {}
        self.doc_list = {}
        self.stop_words = self.build_stop_words_list(stop_words_path)
        self.buildIndex()
        self.num_docs = len(self.doc_list)
        self.terms = list(self.index)
        self.calc_docs_idf()
        self.num_terms = len(self.terms)
        self.term_idx_map = self.build_term_index_mapping()
        self.doc_vec = self.build_vector_space()

        self.DOC_MAG = 'magnitude'
        self.DOC_VEC = 'vector'

    def build_query_vector(self, query_terms):
        for i, word in enumerate(query_terms):
            query_terms[i] = self._tokenize(word)

        query_vec = [0] * self.num_terms
        ## TODO build this

    def build_term_index_mapping(self):
        ## Builds a mapping of terms to their indexes.
        term_idx_map = {
            k:v for v, k in enumerate(self.terms)
        }
        return term_idx_map

    def build_vector_space(self):
        doc_vec_mapping = {}
        for doc in self.doc_list:
            doc_file = open(self.doc_list[doc][0], 'r')
            content = doc_file.read()
            lines = content.split('\n')
            doc_vec = [0] * self.num_terms
            ## Initially, we fill up the doc_vec with term frequency, in the doc
            ## Then, we take it's log. We have the total number of documents. We
            ## have the doc frequency, so we multiply with it's
            for line in lines:
                if len(line) == 0:
                    continue
                words = line.split(' ')
                for word in words:
                    word = self._tokenize(word)
                    if word == '' or word in self.stop_words:
                        continue
                    elif word in self.index:
                        doc_vec[self.term_idx_map[word]] += 1

            magnitude = 0
            for term in self.index:
                idx = self.term_idx_map[term]
                tf = doc_vec[idx]
                doc_vec[idx] = math.log10(1 + tf) * self.index[term][0]
                magnitude += doc_vec[idx] ** 2

            magnitude = math.sqrt(magnitude)

            doc_vec_mapping[doc] = {
                self.DOC_MAG = magnitude,
                self.DOC_VEC = doc_vec
            }
        return doc_vec_mapping

    def calc_docs_idf(self):
        '''
        new_index structure :
        term_1 : [
            idf,
            [doc_1, weight, (pos1, pos2 ...)],
            [doc_2, weight, (pos1, pos2 ...)],
        ],

        '''
        new_index = {}
        for term in self.index:
            df = len(self.index[term])
            idf = math.log10(self.num_docs) - math.log10(df)
            doc_index = {}
            doc_details = [idf]
            for doc in self.index[term]:
                details = []

                ## Add docId
                details.append(doc[0])

                ## Add weight
                tf_d = len(doc[1])
                weight = math.log10(1 + tf_d) * idf
                details.append(weight)
                details.append(tuple(doc[1]))
                doc_details.append(details)
            new_index[term] = doc_details
        self.index = new_index

    def build_stop_words_list(self, stop_words_path):
        stop_file = open(stop_words_path, 'r')
        stop_words = set()
        for line in stop_file.readlines():
            word = line[:-1]
            stop_words.add(word)
        return stop_words

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
            count += 1
            docID = count

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
        return

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
                if word == '' or word in self.stop_words:
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

    def exact_query(self, query_terms, k):
        #function for exact top K retrieval (method 1)
        #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        pass

    def inexact_query_champion(self, query_terms, k):
        #function for exact top K retrieval using champion list (method 2)
        #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        pass

    def inexact_query_index_elimination(self, query_terms, k):
        #function for exact top K retrieval using index elimination (method 3)
        #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        pass

    def inexact_query_cluster_pruning(self, query_terms, k):
        #function for exact top K retrieval using cluster pruning (method 4)
        #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        pass

    def print_dict(self):
        #function to print the terms and posting list in the index
        pass

    def print_doc_list(self):
    # function to print the documents and their document id
        pass
