import math, time

# Helpers
def tuplist(d):
    return [(k, v) for k, v in d.items()]
    
def doc_distinct(index):
    #How many different documents are there?
    docid_set = set()
    indx = tuplist(index)
    for w in indx:
        docid_set = docid_set.union(list(w[1].keys()))
    return docid_set
    
def doc_wordcount(index):
    # how many distinct words exist in an individual document?
    indx = tuplist(index)
    wordcounts_dict = {}
    for w in indx:
        for docid in w[1].keys():
            if docid not in wordcounts_dict:
                wordcounts_dict[docid] = 0
            wordcounts_dict[docid] += 1
    return wordcounts_dict

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index,termWeighting):
        self.total_time = 0
        self.q_count = 0
        self.timed = input("Measure time?").lower() == 'yes'
        if self.timed:
            self.max_q = int(input("How many queries?"))
        """ index         : a dictionary where each key is a word and each
                            value is a dictionary. In each of the
                            sub-dictionaries, each key is a docid from
                            documents.txt and each value is the frequency of
                            the word in the document with said docid.
                    
            termWeighting : the termWeighting scheme
            
            Instantiates a new retrieve object
        """
        #Initialise Variables
        self.index = index
        self.termWeighting = termWeighting
        
        
        """ Since the following variables are the same for a given set of
            documents regardless of the queries, calculate and store them to
            avoid repetition. """
        self.all_docids = doc_distinct(index)
        self.wordcounts = doc_wordcount(index)
        

    # Method to apply query to index
    def forQuery(self,query):
        if self.timed:
            t_start = time.time()
        """ query : a dictionary where each key is a string and each value is a
                    integer
            
            Returns a list of docids
        """
        
        """ get sub-dictionaries in self.index that have a word that exists 
            in the query """
        valid_sub_dict = []
        for q_string in query.keys():
            valid_sub_dict += list(filter(lambda x: x[0]==q_string,
                                          tuplist(self.index)))
        
        hits = dict(zip(self.all_docids, [0]*len(self.all_docids)))
        
        """ For each key (word), add the tfidf score the hits-dictionary."""
        for pair in valid_sub_dict:
            for docid in self.all_docids:
                if docid in pair[1].keys():
                    if self.termWeighting is not "binary":
                        multiplicand = self.tfidf_wordscore(pair, self.termWeighting == "tfidf")[docid]
                    else:
                        multiplicand = 1
                    summand = multiplicand *query[pair[0]]
                    hits[docid] += summand
        
        #Sort hits from highest scoring docid to lowest scoring
        ranked = tuplist(hits)
        ranked.sort(key=lambda tup: tup[1])
        ranked.reverse()
        if self.timed:
            q_time = time.time() - t_start
            self.q_count += 1
            print(self.q_count)
            self.total_time += q_time
            if self.q_count==self.max_q:
                print(self.total_time)
        return list(zip(*ranked))[0]
    
    def tf_wordscore(self, pair):
        normalized = {}
        contained = 0 # how many documents contain the word
        for docid in self.wordcounts.keys():
            if docid not in pair[1]:
                normalized[docid] = 0
            else:
                normalized[docid] = pair[1][docid]/self.wordcounts[docid]
                contained += 1
        return normalized, contained

    def tfidf_wordscore(self,pair, is_idf):
        #Returns a dictionary of docids to tf or tfidf scores for a given word.
    
        #Calculate normalized frequencies of the word in each document
        normalized, contained = self.tf_wordscore(pair)
        
        """ Calculate inverse document frequency, the more common it is among all
            documents, the lower it will be """
        idf = (math.log(len(self.all_docids) / (1 + contained))) if is_idf else 1
        return dict(map(lambda docid: (docid, normalized[docid]*idf), pair[1]))
