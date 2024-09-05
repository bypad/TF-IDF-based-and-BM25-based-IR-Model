# Question 2. TF*IDF-based IR Model

# Import packages and file
import Question_1
import glob, os
import string
from stemming.porter2 import stem
import math

# Open and read the given stopping words list
stopwords_f = open('common-english-words.txt', 'r')
stopwordList = stopwords_f.read().split(',')
stopwords_f.close()


# Task 2.1
def my_df(coll):
    """ Calculate document-frequency (df) for a Rcv1Doc collection """

    df_ = {}
    for doc in coll:
        terms = doc.get_term_list()  # Returns a sorted list of all terms in the document
        for term in terms:
            try:
                df_[term] += 1  # Increase the document frequency when the term occurs again
            except KeyError:
                df_[term] = 1
    return df_


# Task 2.2
def my_tfidf(doc, df, ndocs):
    """ Calculate TF*IDF value (weight) of every term in a Rcv1Doc object """

    tfidf_dict = {}

    terms = doc.get_term_list()  # Returns a sorted list of all terms in the document
    for term in terms:
        tf = math.log(doc.terms.get(term, 0)) + 1  # Calculate Term Frequency (TF)
        idf = math.log(ndocs / df.get(term, 0))  # Calculate Inverse Document Frequency (IDF)
        tfidf = tf * idf  # Compute the TF-IDF value
        tfidf_dict[term] = tfidf

    norm = math.sqrt(sum(tfidf ** 2 for tfidf in tfidf_dict.values()))  # Normalise the TF-IDF values
    for term in tfidf_dict:
        tfidf_dict[term] /= norm  # Divide by the computed norm

    return tfidf_dict


def parse_single_rcvlv2(stop_words, inputpath):
    """ Parse a single document and return a Rcv1Doc object for the document """

    myfile = open(inputpath)
    rcv_doc = Question_1.Rcv1Doc(None, {}, 0)  # Initialise a Rcv1Doc object
    word_count = 0
    start_end = False
    file_ = myfile.readlines()

    for line in file_:
        line = line.strip()
        if (start_end == False):
            if line.startswith("<newsitem "):
                for part in line.split():
                    if part.startswith("itemid="):
                        docid = part.split("=")[1].split("\"")[1]
                        rcv_doc.set_docID(docid)  # Set the document ID
                        break
            if line.startswith("<text>"):  # Tokenise the <text> content
                start_end = True
        elif line.startswith("</text>"):
            break
        else:
            # Remove HTML tags, digits, and punctuation characters
            line = line.replace("<p>", "").replace("</p>", "")
            line = line.translate(str.maketrans('', '', string.digits)).translate(
                str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
            line = line.replace("\\s+", " ")

            for term in line.split():
                word_count += 1  # Words are sequence of characters separated by whitespace
                term = stem(term.lower())  # Terms are words after being stemmed and converted to lowercase
                if len(term) > 2 and term not in stop_words:  # wk3
                    rcv_doc.add_term(term)  # Add term and increase the term frequency

    rcv_doc.set_doc_length(word_count)  # Set the document length

    return rcv_doc


# Task 2.3
def my_ranking_model(query, tfidf):
    """ Calculate a ranking score using the abstract model of ranking for a given query """

    total_score = 0

    for term, frequency in query.items():
        gi = frequency  # Query feature function
        fi = tfidf.get(term, 0)  # Document feature function
        total_score += gi * fi

    return total_score


def main():
    """ Define a main function to test  my_df(), my_tfidf(), and my_ranking_model() """

    with open("PhuongAnhDo_Q2.txt", "a") as file:

        # Testing my_df()
        coll = Question_1.parse_rcvlv2(stopwordList, "RCV1v2/")
        df_ = my_df(coll)
        ndocs = len(coll)
        num_terms = len(df_)
        file.write(f"There are {ndocs} documents in this data set and contains {num_terms} terms \n")
        file.write("The following are the terms’ document-frequency: \n")
        for term, freq in sorted(df_.items(), key=lambda item: item[1], reverse=True):  # Sort by frequency
            file.write(f"{term}: {freq}\n")

        queries = [
            "Reuters French Advertising & Media Digest",
            "15 Palestinians, two Israelis killed in clashes",
            "Great-West Life tops Royal Bank bid for London Ins",
            "Shooting, protests spread in Gaza, West Bank"
        ]
        ranking_results = []

        for query_idx, query in enumerate(queries, 1):
            parse_query_result = Question_1.parse_query(query, stopwordList)
            ranking = []
            for file_path in glob.glob(os.path.join("RCV1v2/", "*.xml")):
                # Testing my_tfidf()
                doc = parse_single_rcvlv2(stopwordList, file_path)
                tfidf_weights = my_tfidf(doc, df_, ndocs)
                docid = doc.get_docID()
                num_term = doc.get_num_terms()
                file.write(f'\nDocument {docid} contains {num_term} terms\n')
                sorted_terms = sorted(tfidf_weights.items(), key=lambda x: x[1], reverse=True)
                for term, weight in sorted_terms[:20]:  # Top 20 terms by the tf*idf weight
                    file.write(f"{term}: {weight}\n")
                # Testing my_ranking_model()
                score = my_ranking_model(parse_query_result, tfidf_weights)
                ranking.append((docid, score))
                file.write("\n")
            ranking_results.append((query, ranking))

        # Write ranking results for all queries
        for query, ranking in ranking_results:
            file.write(f'\nThe Ranking Result for query: {query}\n')
            sorted_ranking = sorted(ranking, key=lambda x: x[1], reverse=True)  # Sort by ranking score
            for docid, score in sorted_ranking:
                file.write(f'{docid}: {score}\n')


if __name__ == "__main__":
    main()