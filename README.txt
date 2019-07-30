Document Indexing:

1. Read each file sequentialy, parse the files line by line.
2. Build an index of term to position mapping for each file.
3. Merge all the indexes of all the files.

Collection Index is a dictionary.

{
    'term1': [
        (doc_id, [position1, position2, ...]),
        (doc_id, [position1, position2, ...]),
        (doc_id, [position1, position2, ...]),
    ],
    'term2': [
        (doc_id, [position1, position2, ...]),
        (doc_id, [position1, position2, ...]),
        (doc_id, [position1, position2, ...]),
    ],
    ... 
}

Vector Space of Documents:

To build the vector space, we again read in all the documents again. 
Since we already have all the terms in the corpus, it is easy to build a vector.
We also build a mapping of term to it's index position. 

