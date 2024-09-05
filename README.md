# TF-IDF-based-and-BM25-based-IR-Model

QUESTION 1:
- Define a document parsing function parse_rcv1v2(stop_words, inputpath) to parse a data collection (e.g., RCV1v2 dataset), where parameter stop_words is a list of common English words (use the file 'common-english-words.txt' to find all stop words), and parameter inputpath is the folder that stores a set of XML files
- Define a query parsing function parse_query(query0, stop_words), where we assume the original query is a simple sentence or a title in a String format (query0), and stop_words is a list of stop words get from 'common-english-words.txt'

QUESTION 2: TF*IDF-based IR Model
- Define a function my_df(coll) to calculate document-frequency (df) for a given Rcv1Doc collection coll and return a {term:df, ...} dictionary
- Use Eq. (1) to define a function my_tfidf(doc, df, ndocs) to calculate TF*IDF value (weight) of every term in a Rcv1Doc object, where doc is a Rcv1Doc object or a dictionary of {term:freq,...}, df is a {term:df, ...} dictionary, and ndocs is the number of documents in a given Rcv1Doc collection. The function returns a {term:tfidf_weight , ...} dictionary for the given document doc

QUESTION 3: BM25-based IR Model
- Define a Python function avg_length(coll) to calculate and return the average document length of all documents in the collection coll
- Use Eq. (3) to define a python function my_bm25(coll, q, df) to calculate documents’ BM25 score for a given original query q, where df is a {term:df, ...} dictionary (call function parse_query() that you defined for Question 1). For the given query q, the function returns a dictionary of {docID: bm25_score, ... } for all documents in collection coll


