Merge algorithm

The merge algorithm in index.py works by merging two lists of document IDs which have already been sorted. 

The basic algorithm to intersect two lists is given below.

INTERSECT(p1, p2) {
	answer <- <>
	l1 = 0
	l2 = 0
	while l1 < len(p1) and l2 < len(p2)
		if docID(l1) = docID(l2) {
			ADD(answer, docID(l1)) 
			l1 += 1 
			l2 += 1 
		} else if docID(l1) < docID(l2) {
			l1 += 1
		} else {
			l2 += 1
		}
	return answer
}

However, the above algorithm does not allow us to merge multiple lists. 
Therefore, the algorithm that I've used is as follows:

query_algo(query) {
	# Clean the query and get them all in same form as tokens
	query = clean(query)

	# Obtain the term frequencies for all the terms in the query
	query_term_freq = get_term_frequency(query) 

	# Sort list of terms based on term frequency in the index
	sorted_terms = sort(quer_term_freq)

	# Get the smallest document list
	doc_list = get_doc_list_for_first_term()

	for eaach term in the rest of the query terms {
		query_doc_list = get_doc_list_for_term(term)
		doc_list = INTERSECT(doc_list, query_doc_list)
	}

	for document in doc_list {
		print document_name
	}
}
