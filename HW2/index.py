#Python 3.x
import re
import os
import collections
import time
import string
import math
import random
#import other modules as needed

class index:
    def __init__(self,path='test_c/', stop_words_path='stop-list.txt'):
        self.path = path
        self.index = {}
        self.doc_list = {}
        s_t = time.time()
        self.stop_words = self.build_stop_words_list(stop_words_path)
        e_t = time.time()
        t_t = e_t - s_t
        print('Time taken to build stop words list : ', t_t)
        s_t = time.time()
        self.buildIndex()
        e_t = time.time()
        t_t = e_t - s_t
        print('Time taken to build index : ', t_t)
        
        s_t = time.time()
        self.num_docs = len(self.doc_list)
        self.terms = list(self.index)
        
        ## calc the idf for each document
        self.calc_docs_idf()
        self.num_terms = len(self.terms)
        
        ## Constant keys for magnitude and vector of documents
        self.DOC_MAG = 'magnitude'
        self.DOC_VEC = 'vector'
        
        ## mapping of term to it's index in vector space
        self.term_idx_map = self.build_term_index_mapping()
        self.doc_vec = self.build_vector_space()
        e_t = time.time()
        t_t = e_t - s_t
        print('Time taken to build vector space : ', t_t)

        s_t = time.time() 
        self.CHAMP_R = 8
        self.champ_list = self.get_champion_list()
        e_t = time.time()
        t_t = e_t - s_t
        print('Time taken to build champion lists', t_t)
        
        ## Cluster Pruning
        s_t = time.time()
        self.leader = self.get_leader_vectors()
        self.cluster = self.get_clusters()
        e_t = time.time()
        t_t = e_t - s_t
        print('Time taken to prune clusters : ', t_t)

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
    
    def cosine_similarity(self, q_vec, q_mag, d_vec, d_mag):
        dot = sum([a * b for a, b in zip(q_vec, d_vec)])
        mag = q_mag * d_mag
        if mag == 0:
            return 0
        return dot/mag
    
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
    
    def rank_docs(self, docs_sim):
        ranked = sorted(docs_sim, key=lambda val : val[1])
        ranked.reverse()
        return ranked

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
    
    def _idf(self, term):
        '''Obtain the Inter-Document Frequency of the given term
        
        :param term: Term
        :type term: string
        :return: IDF value of given term
        :rtype: float
        '''
        return self.index[term][0]

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
        # return new_index

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
            word = line[:-1]
            stop_words.add(word)
        stop_file.close()
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
    
    def get_champion_list(self):
        ## For each term, keep track of the top 8 docs.
        champ_vec = dict()
        doc_ids = set()
        for term in self.index:
            c_list = []
            docs = self.index[term][1:]
            
            # append idf to head of list
            c_list.append(self.index[term][0])
            if len(docs) < self.CHAMP_R:
                for doc in docs:
                    doc_ids.add(doc[0])
            else:
                # sort and obtain the top R documents
                docs = sorted(docs, key=lambda val: val[1])
                docs.reverse()
                for i in range(self.CHAMP_R):
                    doc_ids.add(docs[i][0])
        for doc_id in doc_ids:
            champ_vec[doc_id] = self.doc_vec[doc_id]
        return champ_vec
    
    def get_leader_vectors(self):
        doc_ids = list(self.doc_list)
        num_leaders = int(math.sqrt(self.num_docs))
        leader_docs = random.choices(doc_ids, k=num_leaders)
        return leader_docs

    def get_clusters(self):
        leader_vec = {}
        clusters = {}
        for doc in self.leader:
            leader_vec[doc] = self.doc_vec[doc]
            clusters[doc] = []
        for doc in self.doc_vec:
            d_mag = self.doc_vec[doc][self.DOC_MAG]
            d_vec = self.doc_vec[doc][self.DOC_VEC]
            if doc in leader_vec:
                continue
            max_sim = -1
            max_leader = -1
            for leader in self.leader:
                l_mag = self.doc_vec[leader][self.DOC_MAG]
                l_vec = self.doc_vec[leader][self.DOC_VEC]
                sim = self.cosine_similarity(d_vec, d_mag, l_vec, l_mag)
                if sim > max_sim:
                    max_sim = sim
                    max_leader = leader
            clusters[max_leader].append(doc)
        return clusters
        
    def rank_leaders(self, q_vec, q_mag):
        ranked_leaders = []
        for leader in self.leader:
            l_mag = self.doc_vec[leader][self.DOC_MAG]
            l_vec = self.doc_vec[leader][self.DOC_VEC]
            sim = self.cosine_similarity(q_vec, q_mag, l_vec, l_mag)
            ranked_leaders.append((leader, sim))
        ranked_leaders = sorted(ranked_leaders, key=lambda val : val[1])
        ranked_leaders.reverse()
        return ranked_leaders
    
    def get_ranked_docs(self, q_vec, q_mag, leader_id, cluster):
        ranked_docs = []
        l_mag = self.doc_vec[leader_id][self.DOC_MAG]
        l_vec = self.doc_vec[leader_id][self.DOC_VEC]
        sim = self.cosine_similarity(q_vec, q_mag, l_vec, l_mag)
        ranked_docs.append((leader_id, sim))
        for doc in cluster:
            d_mag = self.doc_vec[doc][self.DOC_MAG]
            d_vec = self.doc_vec[doc][self.DOC_VEC]
            sim = self.cosine_similarity(q_vec, q_mag, d_vec, d_mag)
            ranked_docs.append((doc, sim))
        ranked_docs = sorted(ranked_docs, key=lambda val: val[1])
        ranked_docs.reverse()
        return ranked_docs
        
    def get_top_k_docs(self, q_vec, q_mag, k):
        ## rank leaders in order of similarity to query
        ranked_leaders = self.rank_leaders(q_vec, q_mag)
        
        # k_doc has a structure of [(doc_id, similarity), ]
        k_docs = []
        for leader in ranked_leaders:
            ## rank cluster documents and leader document on similarity to query
            ## leader[0] is just the id
            ranked_docs = self.get_ranked_docs(q_vec, q_mag, leader[0], self.cluster[leader[0]])
            k_docs.extend(ranked_docs)
        return k_docs
        
    def exact_query(self, query_terms, k):
        #function for exact top K retrieval (method 1)
        #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        s_t = time.time()
        q_vec, q_mag = self.build_query_vector(query_terms)
        docs_sim = self.calc_sim_docs(q_vec, q_mag, self.doc_vec)
        ranked_sim = self.rank_docs(docs_sim)
        e_t = time.time()
        print('Time taken : ', e_t - s_t)
        for i in range(k):
            doc_id, sim_score = ranked_sim[i]
            doc_name = self.doc_list[doc_id]
            print('Document : ', doc_name, ' Similarity : ', sim_score)

    def inexact_query_champion(self, query_terms, k):
        #function for exact top K retrieval using champion list (method 2)
        #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        s_t = time.time()
        q_vec, q_mag = self.build_query_vector(query_terms)
        
        ## Difference betweeen this and the line above is that we're passing in champ_list to calc_sim_docs
        docs_sim = self.calc_sim_docs(q_vec, q_mag, self.champ_list)
        ranked_sim = self.rank_docs(docs_sim)
        e_t = time.time()
        print('Time taken : ', e_t - s_t)
        for i in range(k):
            doc_id, sim_score = ranked_sim[i]
            doc_name = self.doc_list[doc_id]
            print('Document : ', doc_name, ' Similarity : ', sim_score)

    def inexact_query_index_elimination(self, query_terms, k):
        #function for exact top K retrieval using index elimination (method 3)
        #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        s_t = time.time()
        q_terms = []
        for i, term in enumerate(query_terms):
            term = self._tokenize(term)
            if term not in self.stop_words:
                query_terms[i] = term
                idf = self._idf(term)
                q_terms.append((term, idf))
        q_terms = sorted(q_terms, key=lambda val : val[1])
        q_terms.reverse()
        num_terms = len(q_terms)
        if num_terms % 2 == 0:
            q_terms = q_terms[:num_terms // 2]
        else:
            q_terms = q_terms[:num_terms // 2 + 1]
        
        q_terms = [val[0] for val in q_terms]
            
        q_vec, q_mag = self.build_query_vector(q_terms)
        docs_sim = self.calc_sim_docs(q_vec, q_mag, self.champ_list)
        ranked_sim = self.rank_docs(docs_sim)
        e_t = time.time()
        print('Time taken : ', e_t - s_t)
        for i in range(k):
            doc_id, sim_score = ranked_sim[i]
            doc_name = self.doc_list[doc_id]
            print('Document : ', doc_name, ' Similarity : ', sim_score)

    def inexact_query_cluster_pruning(self, query_terms, k):
        #function for exact top K retrieval using cluster pruning (method 4)
        #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        
        ## leader_vec is a list of doc_ids of the 
        
        ## clusters is a dict with the following structure 
        ## { leader_doc_id : [cluster_members_doc_id]}
        
        s_t = time.time()
        q_vec, q_mag = self.build_query_vector(query_terms)

        similarity = self.get_top_k_docs(q_vec, q_mag, k)
        
        ranked_sim = sorted(similarity, key=lambda val : val[1])
        ranked_sim.reverse()
        e_t = time.time()
        print('Time taken : ', e_t - s_t)
        
        for i in range(k):
            print('Document : ', ranked_sim[i][0], ' Similarity : ', ranked_sim[i][1])

    def print_dict(self):
        #function to print the terms and posting list in the index
        print('Printing Index')
        for i in self.index:
            print(i)

    def print_doc_list(self):
    # function to print the documents and their document id
        for i in self.index:
            print(self.index[i])
    
    def clean_query(self, query):
        print('Query Terms')
        q_terms = query.split(' ')
        q = []
        for term in q_terms:
            if term not in self.stop_words:
                q.append(term)
        print(q)
        return q

if __name__ == '__main__':
    ir = index('collection')
    
    query_terms = ir.clean_query('US and december without india reconsider')
    ir.exact_query(query_terms, 10)
    ir.inexact_query_champion(query_terms, 10)
    ir.inexact_query_index_elimination(query_terms, 10)
    ir.inexact_query_cluster_pruning(query_terms, 10)

    query_terms = ir.clean_query('Ahead of the family matter')
    ir.exact_query(query_terms, 10)
    ir.inexact_query_champion(query_terms, 10)
    ir.inexact_query_index_elimination(query_terms, 10)
    ir.inexact_query_cluster_pruning(query_terms, 10)
    
    query_terms = ir.clean_query('is not needless be running')
    ir.exact_query(query_terms, 10)
    ir.inexact_query_champion(query_terms, 10)
    ir.inexact_query_index_elimination(query_terms, 10)
    ir.inexact_query_cluster_pruning(query_terms, 10)
    
    query_terms = ir.clean_query('corruption and rivalries')
    ir.exact_query(query_terms, 10)
    ir.inexact_query_champion(query_terms, 10)
    ir.inexact_query_index_elimination(query_terms, 10)
    ir.inexact_query_cluster_pruning(query_terms, 10)
    
    query_terms = ir.clean_query('India and US are countries')
    ir.exact_query(query_terms, 10)
    ir.inexact_query_champion(query_terms, 10)
    ir.inexact_query_index_elimination(query_terms, 10)
    ir.inexact_query_cluster_pruning(query_terms, 10)