import math

# Helpers
def tuplist(d):
    return [(k, v) for k, v in d.items()]

def tf(word, doc_indices):
    #Frequency of a word in one document
    return doc_indices.words.count(word) / len(doc_indices.words)

def idf(s, doc_indiceslist):
    #how common does string s appear in every document being analyzed
    return math.log(len(doc_indiceslist) / (1 + n_containing(s, doc_indiceslist)))

def n_containing(word, doc_indiceslist):
    #Number of distinct documents that contain the word
    return sum(1 for doc_indices in doc_indiceslist if word in doc_indices.words)

def formula_tfidf(word, doc_indices, doc_indiceslist):
    #tdf score is product of tf and idf
    return tf(word, doc_indices) * idf(word, doc_indiceslist)

def tfidf_wordscore(pair, wordcounts_dict, index):
    #Returns a dictionary of docids to tfidf scores for a given word.

    #Calculate normalized frequencies of the word in each document
    normalized = {}
    n_containing = 0 # how many documents contain the word
    for docid in wordcounts_dict.keys():
        if docid not in pair[1]:
            normalized[docid] = 0
        else:
            normalized[docid] = pair[1][docid]/wordcounts_dict[docid]
            n_containing += 1
    """ Calculate inverse document frequency, the more common it is among all
        documents, the lower it will be """
    idf = math.log(len(index) / (1 + n_containing))
    wordscore = {}
    for docid in pair[1]:
        wordscore[docid] = normalized[docid]*idf #product of tf and idf
    return wordscore
    
def doc_distinct(index):
    #How many different documents are there?
    docid_set = set()
    indx = tuplist(index)
    for w in indx:
        docid_set = docid_set.union(list(w[1].keys()))
    return docid_set
    
def doc_wordcount(index):
    """ doc_wordcount:  a dictionary telling us how many distinct words exist
                        for a single document. Keys are docids.
    """
    indx = tuplist(index)
    wordcounts_dict = {}
    for w in indx:
        for docid in w[1].keys():
            #Does uniqueness affect final evaluation?
            if docid not in wordcounts_dict:
                wordcounts_dict[docid] = 0
            wordcounts_dict[docid] += w[1][docid] #Either += 1 of w[1][docid]
    return wordcounts_dict

class Retrieve:
    # Constructors
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index,termWeighting):
        """ index         : a dictionary where each key is a word and each
                            value is a dictionary. In each of the
                            sub-dictionaries, each key is a docid from
                            documents.txt and each value is the frequency of
                            the word in the document with said docid.
                    
            termWeighting : the termWeighting scheme
            
            Instantiates a new retrieve object
        """
        #Initialise Variables, usually doesn't change
        self.index = index
        self.termWeighting = termWeighting
        
        
        """ Since the following variables are the same for a given set of
            documents regardless of the queries, calculate and store them to
            avoid repetition. """
        self.all_docids = doc_distinct(index)
        self.wordcounts = doc_wordcount(index)
        

    # Method to apply query to index
    def forQuery(self,query):
        """ query : a dictionary where each key is a string and each value is a
                    integer
            
            Returns a list of docids
        """
        #Query changes, tfidf, tf and binary by default.
        
        return self.tfidf(query)
    
    # termWeighting schemes
    def tfidf(self, query):
        """ Words that appear frequently in one document score higher.
            Words that appear in many documents score lower.
            
            The document score is sum of each of its word scores.
        """
        
        #Use wordcounts of each document
        
        """ get sub-dictionaries in self.index that have a word that exists 
            in the query """
        valid_sub_dict = []
        for q_string in query.keys():
            valid_sub_dict += list(filter(lambda x: x[0]==q_string,
                                          tuplist(self.index)))
        
        #hit-dictionary
        #all_docid = set().union(*list(map(lambda pair: pair[1].keys(),
        #                              valid_sub_dict)))
        hits = dict.fromkeys(self.all_docids)
        
        """ For each key (word), add the tfidf score the hits-dictionary."""
        for pair in valid_sub_dict:
            for docid in self.all_docids:
                if hits[docid] is None:
                    hits[docid] = 0
                if docid in pair[1].keys():
                    multiplicand = tfidf_wordscore(pair, self.wordcounts, self.index)[docid]
                    summand = multiplicand *query[pair[0]]
                    hits[docid] += summand
        
        #Sort hits from highest scoring docid to lowest scoring
        ranked = tuplist(hits)
        ranked.sort(key=lambda tup: tup[1])
        ranked.reverse()
        return list(zip(*ranked))[0]
    
    def tf(self, query):
        """ get sub-dictionaries in self.index that have a word that exists 
            in the query """
        valid_sub_dict = []
        for q_string in query.keys():
            valid_sub_dict += list(filter(lambda x: x[0]==q_string,
                                          tuplist(self.index)))
        
        #hit-dictionary
        all_docid = set().union(*list(map(lambda pair: pair[1].keys(),
                                      valid_sub_dict)))
        hits = dict.fromkeys(all_docid)
        
        """ For each key, add the product of document frequency to query
            frequency to the hits-dictionary."""
        for pair in valid_sub_dict:
            for docid in all_docid:
                if docid in pair[1].keys():
                    if hits[docid] is None:
                        hits[docid] = 0
                    multiplicand = pair[1][docid]
                    summand = multiplicand *query[pair[0]]
                    hits[docid] += summand
        
        #Sort hits from highest scoring docid to lowest scoring
        ranked = tuplist(hits)
        ranked.sort(key=lambda tup: tup[1])
        ranked.reverse()
        return list(zip(*ranked))[0]
    