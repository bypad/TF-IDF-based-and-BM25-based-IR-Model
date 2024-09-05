# Question 1. Parsing of Documents & Queries

# Import packages
import glob, os
import string
from stemming.porter2 import stem

# Open and read the given stopping words list
stopwords_f = open('common-english-words.txt', 'r')
stopwordList = stopwords_f.read().split(',')
stopwords_f.close()


class Rcv1Doc:
    """ A Rcv1Doc class to represent a document """

    def __init__(self, docID, terms, doc_len):
        self.docID = docID
        self.terms = {}
        self.doc_len = doc_len

    def set_docID(self, docID):  # Sets the document ID
        self.docID = docID

    def get_docID(self):  # Returns the document ID
        return self.docID

    def set_doc_length(self, length):  # Sets the document length
        self.doc_len = length

    def get_doc_length(self):  # Returns the document length
        return self.doc_len

    def add_term(self, term):  # Increases the term frequency when the term occurs again
        self.terms[term] = self.terms.get(term, 0) + 1

    def get_term_list(self):  # Returns a sorted list of all terms
        return sorted(self.terms.keys())

    def get_num_terms(self):  # Returns the total number of terms
        return len(self.terms)


# Task 1.1
def parse_rcvlv2(stop_words, inputpath):
    """ Parse a data collection (e.g., RCV1v2 dataset) and return the collection of Rcv1Doc objects """

    coll = []

    for file_path in glob.glob(os.path.join(inputpath, "*.xml")):
        rcv_doc = Rcv1Doc(None, {}, 0)  # Initialise a Rcv1Doc object for each document
        word_count = 0
        start_end = False

        with open(file_path, 'r') as myFile:
            for line in myFile:
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
                        if len(term) > 2 and term not in stop_words:
                            rcv_doc.add_term(term)  # Add term and increase the term frequency

        rcv_doc.set_doc_length(word_count)  # Set the document length
        coll.append(rcv_doc)

    return coll


# Task 1.2
def parse_query(query0, stop_words):
    """ Parse a given query """

    curr_querry = {}

    # Remove HTML tags, digits, and punctuation characters
    query0 = query0.replace("<p>", "").replace("</p>", "")
    query0 = query0.translate(str.maketrans('', '', string.digits)).translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    query0 = query0.replace("\\s+", " ")

    for term in query0.split():
        term = stem(term.lower())  # Stem and covert to lowercase
        if len(term) > 2 and term not in stop_words:
            curr_querry[term] = curr_querry.get(term, 0) + 1  # Increase the term frequency

    return curr_querry


# Task 1.3
def main():
    """ Define a main function to test function parse_rcv1v2( ) and parse_query( ) """

    with open("PhuongAnhDo_Q1.txt", "w") as f:

        # Test parse_rcv1v2() function
        coll = parse_rcvlv2(stopwordList, "RCV1v2/")
        for doc in coll:
            docid = doc.get_docID()
            num_of_terms = doc.get_num_terms()
            doc_len = doc.get_doc_length()
            f.write(f"\nDocument {docid} contains {num_of_terms} terms and has total of {doc_len} words \n")
            for term, freq in sorted(doc.terms.items(), key=lambda item: item[1], reverse=True):  # Sort by frequency
                f.write(f"{term}: {freq}\n")

        # Test parse_query() function
        sample_query = 'CANADA: Sherritt to buy Dynatec, spin off unit, canada.'
        parsed_query = parse_query(sample_query, stopwordList)
        parsed_query_str = str(parsed_query)
        f.write("\n")
        f.write(f'Query: {sample_query} \n')
        f.write("The parsed query:\n")
        f.write(f'{parsed_query_str}\n')


if __name__ == "__main__":
    main()