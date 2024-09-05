# Question 3. BM25-based IR Model

# Import packages and files
import math
import Question_1
import Question_2

# Open and read the given stopping words list
stopwords_f = open('common-english-words.txt', 'r')
stopwordList = stopwords_f.read().split(',')
stopwords_f.close()

# Task 3.1
def avg_length(coll):
    """ Calculate and return the average document length of all documents """

    total_doc_length = 0
    num_docs = len(coll)

    for doc in coll:
        total_doc_length += doc.get_doc_length()
    avg_doc_length = total_doc_length / num_docs

    return avg_doc_length


# Task 3.2
def my_bm25(coll, q, df):
    """ Calculate documents’ BM25 score for a given original query """

    query_terms = Question_1.parse_query(q, stopwordList)  # Parse the given query
    bm25_scores = {}
    length = {}
    k1 = 1.2
    k2 = 100
    b = 0.75
    N = len(coll)
    R = ri = 0
    avg_dl = avg_length(coll)

    for doc in coll:
        score = 0
        doc_id = doc.get_docID()
        doc_length = doc.get_doc_length()
        K = k1 * ((1 - b) + b * (doc_length / avg_dl))
        for term, freq in query_terms.items():
            fi = doc.terms.get(term, 0)  # Calculate the term frequency in the document
            qfi = freq  # Calculate the term frequency in the query
            ni = df.get(term, 0)  # Calculate the number of documents containing the term

            first_term = ((ri + 0.5) / (R - ri + 0.5)) / ((ni - ri + 0.5) / (N - ni - R + ri + 0.5))
            second_term = ((k1 + 1) * fi) / (K + fi)
            third_term = ((k2 + 1) * qfi) / (k2 + qfi)

            score += math.log(first_term) * second_term * third_term  # Calculate BM25 score

        bm25_scores[doc_id] = score
        length[doc_id] = doc_length

    return bm25_scores, length  # Return the BM25 score and the document length


# Task 3.3
def main():
    """ Define a main function to test my_bm25() """
    
    with open("PhuongAnhDo_Q3.txt", "a") as file:
        # Test avg_length()
        coll = Question_1.parse_rcvlv2(stopwordList, "RCV1v2/")
        average_length = avg_length(coll)
        file.write(f"Average document length for this collection is: {average_length}\n")

        # Test my_bm25()
        df_ = Question_2.my_df(coll)
        queries = ["The British-Fashion Awards", "Rocket attacks", "Broadcast Fashion Awards", "stock market"]
        for query_idx, query in enumerate(queries, 1):
            bm25_scores, length = my_bm25(coll, query, df_)
            file.write(f'\nThe query is: {query}\n')
            file.write("The following are the BM25 score for each document:\n")
            for doc_id, score in bm25_scores.items():
                doc_length = length[doc_id]
                file.write(f'Document ID: {doc_id}, Doc Length: {doc_length} -- BM25 Score: {score} \n')

            # Test my_bm25() for top-6 documents
            file.write(f'\nFor query {query}, the top-6 possible relevant documents are:\n')
            sorted_documents = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)  # Sort by ranking score
            for i, (doc_id, score) in enumerate(sorted_documents[:6], 1):
                file.write(f'{doc_id} {score} \n')


if __name__ == "__main__":
    main()