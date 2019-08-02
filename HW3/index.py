#Python 3.0
import re
import os
import collections
import time
import string
import math

from copy import copy
#import other modules as needed

class index: 
    def __init__(self,path='./doc', doc_id_file='./time/TEXTids.txt', stop_words_path='./time/TIME.STP'): 
        self.path = path
        self.index = {}
        self.doc_list = {}
        self.stop_words = self.build_stop_words_list(stop_words_path)
        self.doc_id_mapping = self.build_doc_id_mapping(doc_id_file)
        self.buildIndex()
        
        self.num_docs = len(self.doc_list)
        self.terms = list(self.index)
        
        ## Calc the idf for each document
        print('Calculating IDF')
        self.calc_docs_idf()
        print('Calculated IDF')
        self.num_terms = len(self.terms)
        
        ## Constant keys for magnitude and vector of documents
        self.DOC_MAG = 'magnitude'
        self.DOC_VEC = 'vector'
    
        ## Constants for vector operations
        self.ADD_OP = 1
        self.SUB_OP = -1
        
        ## Mapping of term to it's index in vector space
        self.term_idx_map = self.build_term_index_mapping()
        
        ## Invert dictionary to map index to term
        self.idx_term_map = dict(map(reversed, self.term_idx_map.items()))
        
        ## Build vector space
        print('Building Vector Space')
        self.doc_vec = self.build_vector_space()
        print('Vector Space Built')
    
    def rank_docs(self, docs_sim):
        ranked = sorted(docs_sim, key=lambda val : val[1])
        ranked.reverse()
        return ranked
    
    def calc_sim_docs(self, q_vec, q_mag, doc_vec):
        similarity = []
        for doc in doc_vec:
            d_mag = doc_vec[doc][self.DOC_MAG]
            d_vec = doc_vec[doc][self.DOC_VEC]
            if d_mag == 0:
                continue
            sim = self.cosine_similarity(q_vec, q_mag, d_vec, d_mag)
            similarity.append((doc, sim))
        return similarity
    
    def cosine_similarity(self, q_vec, q_mag, d_vec, d_mag):
        '''Calculate the cosine similarity of a document and a query
        
        :param q_vec: Query Vector
        :type q_vec: List of floats
        :param q_mag: Magnitude of query vector
        :type q_mag: Float
        :param d_vec: Document Vector
        :type d_vec: List of floats
        :param d_mag: Magnitude of document vector
        :type d_mag: Float
        :return: Similarity score
        :rtype: Float
        '''
        dot = sum([a * b for a, b in zip(q_vec, d_vec)])
        mag = q_mag * d_mag
        if mag == 0:
            return 0
        return dot/mag
    
    def build_term_index_mapping(self):
        '''Maps given term to it's position on vector
        
        :return: Index of term in vector
        :rtype: int
        '''
        ## Builds a mapping of terms to their indexes.
        term_idx_map = {
            k:v for v, k in enumerate(self.terms)
        }
        return term_idx_map
    
    def build_vector_space(self):
        '''Build vector representations of all the documents
        The vector representation is 
        {
            doc_id_1(int) : {
                self.DOC_MAG(magnitude) : magnitude of vector,
                self.DOC_VEC(vector) : document vector
            }
        }
        
        :return: Vector Space of docs
        :rtype: Dictionary
        '''
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
                if tf == 0:
                    val = 0
                else:
                    val = 1 + math.log10(tf)
                doc_vec[idx] = val * self._idf(term)
                magnitude += doc_vec[idx] ** 2

            magnitude = math.sqrt(magnitude)

            doc_vec_mapping[doc] = {
                self.DOC_MAG : magnitude, 
                self.DOC_VEC : doc_vec
            }
            doc_file.close()
        return doc_vec_mapping
    
    def _idf(self, term):
        '''Obtain the Inter-Document Frequency of the given term
        
        :param term: Term
        :type term: string
        :return: IDF value of given term
        :rtype: float
        '''
        return self.index[term][0]
    
    def build_query_vector(self, query_terms):
        '''Builds vector representation of given query
        
        :param query_terms: Terms in the query
        :type query_terms: List of strings
        :return: Vector representation of the query
        :rtype: List of floats
        '''
        for i, word in enumerate(query_terms):
            query_terms[i] = self._tokenize(word)

        query_vec = [0] * self.num_terms
        for term in query_terms:
            if term in self.stop_words:
                continue
            idx = self.term_idx_map[term]
            query_vec[idx] += 1
        
        magnitude = 0
        for term in set(query_terms):
            idx = self.term_idx_map[term]
            idf = self._idf(term)
            tf = query_vec[idx]
            if tf == 0:
                val = 0
            else:
                val = 1 + math.log10(tf)
            val = val * idf
            magnitude += val ** 2
            query_vec[idx] = val
        magnitude = math.sqrt(magnitude)
        return query_vec, magnitude
    
    def build_query_vector(self, query_terms):
        '''Builds vector representation of given query
        
        :param query_terms: Terms in the query
        :type query_terms: List of strings
        :return: Vector representation of the query
        :rtype: List of floats
        '''
        for i, word in enumerate(query_terms):
            query_terms[i] = self._tokenize(word)

        query_vec = [0] * self.num_terms
        for term in query_terms:
            if term in self.stop_words:
                continue
            idx = self.term_idx_map[term]
            query_vec[idx] += 1
        
        magnitude = 0
        for term in set(query_terms):
            idx = self.term_idx_map[term]
            idf = self._idf(term)
            tf = query_vec[idx]
            if tf == 0:
                val = 0
            else:
                val = 1 + math.log10(tf)
            val = val * idf
            magnitude += val ** 2
            query_vec[idx] = val
        magnitude = math.sqrt(magnitude)
        return query_vec, magnitude
    
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
            idf = math.log10(self.num_docs / df)
            doc_index = {}
            doc_details = [idf]
            for doc in self.index[term]:
                details = []

                ## Add docId
                details.append(doc[0])

                ## Add weight
                tf_d = len(doc[1])
                if tf_d == 0:
                    weight = 0
                else:
                    weight = (1 + math.log10(tf_d)) * idf
                details.append(weight)
                details.append(tuple(doc[1]))
                doc_details.append(details)
            new_index[term] = doc_details
        self.index = new_index
    
    def build_doc_id_mapping(self, doc_id_file):
        '''Build document name to document id mapping
        
        :param doc_id_file: File containing document id mappings
        :type doc_id_file: Path to file
        :return: Dictionary <doc_name, doc_id>
        :rtype: Dictionary
        '''
        id_file = open(doc_id_file, 'r')
        
        # Ignoring the first line
        id_file.readline()
        lines = id_file.readlines()
        id_map = {}
        for line in lines:
            tkn = line.split()
            
            # Mapping to -1 because 
            id_map['Doc_' + str(int(tkn[2])) + '.txt'] = int(tkn[0][:-1])
        return id_map
        
    def build_stop_words_list(self, stop_words_path):
        '''Build stop words list
        
        :param stop_words_path: Path to stop words file
        :type stop_words_path: String
        :return: Set of stop words to ignore
        :rtype: Set
        '''
        stop_file = open(stop_words_path, 'r')
        stop_words = set()
        for line in stop_file.readlines():
            if len(line) > 0:
                word = line[:-1]
                stop_words.add(word)
        stop_file.close()
        return stop_words
        
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
        doc_file = open(doc, 'r', encoding='utf-8')
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
        doc_file.close()
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
    
    def clean_query(self, query):
        '''Removes stop words from the given query
        
        :param query: Query terms
        :type query: List of strings
        :return: Cleaned query terms
        :rtype: List of strings
        '''
        print('Query Terms')
        q_terms = query.split(' ')
        q = []
        for term in q_terms:
            if term not in self.stop_words:
                q.append(term)
        return q
    
    def mult(self, value, vector):
        '''Perform scalar multiplication
        
        :param value: Value to be multiplied with vector
        :type value: Float
        :param vector: Vector
        :type vector: Float
        :return: Scalar multiplied vector
        :rtype: List of floats
        '''
        ## Will perform changes in place. But returning nonetheless.
        for i, val in enumerate(vector):
            vector[i] = value * val
        return vector
    
    def _op_vec(self, v1, v2, op):
        '''Perform element wise vector operation. Adds if op is positive. Subtracts otherwise.
        
        :param v1: Vector 1
        :type v1: List of floats
        :param v2: Vector 2
        :type v2: List of floats
        :param op: Operation type
        :type op: Int
        :return: New Vector
        :rtype: List of floats
        '''
        res = []
        for i in range(len(v1)):
            if op >= self.ADD_OP:
                res.append(v1[i] + v2[i])
            elif op <= self.SUB_OP:
                res.append(v1[i] - v2[i])
        return res
    
    def op_vectors(self, doc_ids, op):
        '''Performs Vector operation on a list of vectos
        
        :param doc_ids: List of doc ids whose vectors need to be added
        :type doc_ids: List of int
        :param op: Type of vector operation
        :type op: Constant int
        '''
        if len(doc_ids) == 1:
            return self.doc_vec[doc_ids[0]]
        
        s_vec = copy(self.doc_vec[doc_ids[0]])
        for doc_id in doc_ids[1:]:
            s_vec = self._op_vec(s_vec, self.doc_vec[doc_id], op)
        return s_vec
        
    def _rocchio(self, q_vec, pos_feedback, neg_feedback, alpha, beta, gamma):
        '''Perform Rocchio Query Correction
        
        :param q_vec: Query Vector
        :type q_vec: List of floats
        :param pos_feedback: Relevant Documents
        :type pos_feedback: List of document IDs
        :param neg_feedback: Irrelevant Documents
        :type neg_feedback: List of Document IDs
        :param alpha: Alpha co-ef
        :type alpha: Float
        :param beta: Beta co-ef
        :type beta: Float
        :param gamma: Gamma co-ef
        :type gamma: Float
        '''
        # First term in Rocchio equation
        p1 = self.mult(alpha, q_vec)
        
        beta = beta * (1 / len(pos_feedback))
        p2 = self.op_vectors(pos_feedback, self.ADD_OP)
        p2 = self.mult(beta, p2)
        
        gamma = gamma * (1 / len(neg_feedback))
        p3 = self.op_vectors(neg_feedback, self.ADD_OP)
        p3 = self.mult(gamma, p3)
        
        q_m = self._op_vec(p1, p2, self.ADD_OP)
        q_m = self._op_vec(q_m, p3)
        return q_m
    
    def rebuild_query_terms(self, query_vec):
        '''Obtain query terms given query vector.
        
        :param query_vec: Query Vector
        :type query_vec: List of floats
        :return: Query Terms
        :rtype: List of strings
        '''
        q_terms = []
        for i, val in enumerate(query_vec):
            if val > 0:
                term = self.idx_term_map[i]
                q_terms.append(term)
        return q_terms
        
    def buildIndex(self):
        '''Builds an index of text files in a given directory.
        '''
        # function to read documents from collection, tokenize and build the index with tokens
        # implement additional functionality to support relevance feedback
        # use unique document integer IDs
        print('Building index')
        files = os.scandir(self.path)
        for doc_file in files:
            docID = self.doc_id_mapping[doc_file.name]

            if docID not in self.doc_list:
                self.doc_list[docID] = (doc_file.path, doc_file.name)

            doc_index = self.build_inverted_index_for_doc(doc_file)
            self._add_doc_index_to_dir_index(docID, doc_index)
        print('Index built')

    def rocchio(self, query_terms, pos_feedback, neg_feedback, alpha=1, beta=0.75, gamma=0.15):
        # function to implement rocchio algorithm
        # pos_feedback - documents deemed to be relevant by the user
        # neg_feedback - documents deemed to be non-relevant by the user
        # Return the new query  terms and their weights
        query_terms = self.clean_query(query_terms)
        q_vec, q_mag = self.build_query_vector(query_terms)
        q_modified = self._rocchio(q_vec, pos_feedback, neg_feedback, alpha, beta, gamma)
        q_mod_terms = self.rebuild_query_terms(q_modified)
        return q_mod_terms, q_modified
    
    def query(self, query_terms, k):
        #function for exact top K retrieval (method 1)
        #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        q_vec, q_mag = self.build_query_vector(query_terms)
        docs_sim = self.calc_sim_docs(q_vec, q_mag, self.doc_vec)
        ranked_sim = self.rank_docs(docs_sim)
        for i in range(k):
            doc_id, sim_score = ranked_sim[i]
            doc_name = self.doc_list[doc_id]
            print('Document : ', doc_name, ' Similarity : ', sim_score)
    
    def print_dict(self):
        #function to print the terms and posting list in the index
        print(self.index)

    def print_doc_list(self):
        # function to print the documents and their document id
        print(self.doc_list)