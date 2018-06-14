import csv
import sys
import nltk
from math import log

EMPTY_LIST  = []
EMPTY_SET = {}
TITLE = 1
SUBTITLE = 2
CONTENT = 3
URL = 4
ID = 5
class inverted_index:
    
    def __init__(self):
        self.invertedIndex = {}
        self.articles = {}
        rawData = self.readData()
        self.TOTAL_DOCUMENTS = len(rawData)
        self.buildInvertedIndex(rawData)
    
    def readData(self):
        """"Obtains the data from the csv file and returns it as a 
        list of articles, each of them containing title, content and id"""
        data = []
        with open('estadao_noticias_eleicao.csv', 'rt') as csvFile:
            text = [line for line in csv.reader(csvFile, delimiter=',', quotechar='"')]
        return text


    def buildInvertedIndex(self, rawData):
        for article in rawData:
            id = article[ID]
            self.articles[id] = ["placeholder",article[TITLE], article[SUBTITLE], article[CONTENT], article[URL]]
            self.articles[id]
            self.addEntriesToIndex(article[TITLE], id)
            self.addEntriesToIndex(article[CONTENT], id)
        return self.invertedIndex


    def addEntriesToIndex(self, newsString, articleId):
        """Processes a string extracting the individual words from it 
        and adds <word, list(articleIds)> as a <key, value> pair in the HashMap"""
        for word in newsString.split():
            if (word in self.invertedIndex) and (articleId in self.invertedIndex[word]):
                self.invertedIndex[word][articleId] += 1
            elif (word in self.invertedIndex) and not (articleId in self.invertedIndex[word]):
                self.invertedIndex[word][articleId] = 1
            else:
                self.invertedIndex[word] = {articleId: 1}

    def get_tf(self, dId, query):
        return sum(1 for w in nltk.word_tokenize(query) if dId in self.invertedIndex.get(w, EMPTY_SET).keys())


    def binSearch(self, query):
        # matchedDocs = [ elem \
        #                 for w in nltk.word_tokenize(query) \
        #                 for elem in list(self.invIndex[w].keys()) \
        #                 if w in self.invIndex ]

        matchedDocs = None
        for w in nltk.word_tokenize(query):
            if w in self.invertedIndex:
                matches = set(self.invertedIndex[w].keys())
                matchedDocs = matches if matchedDocs == None else matchedDocs.union(matches)
        # print('docs = {0}'.format(matchedDocs))
        weightedDocs = [(docId, self.get_tf(docId, query)) for docId in matchedDocs]
        return [e[0] for e in sorted(weightedDocs, key= lambda x: x[1], reverse = True)]

    def get_tf_idf(self, dId, query):
        out = 0
        for w in nltk.word_tokenize(query):
            if dId in self.invertedIndex.get(w, EMPTY_SET).keys():
                tf = self.invertedIndex[w][dId]
                idf = log( self.TOTAL_DOCUMENTS / tf )
                out += query.count(w)*tf*idf
        return out

    def tfIdfSearch(self, query):
        docs = self.binSearch(query)
        docs = [(docId, self.get_tf_idf(docId, query)) for docId in docs ]
        return [e[0] for e in sorted(docs, key= lambda x: x[1], reverse = True)]

    def searchOne(self, term):
        """Searches for a specific term in the index and
        returns a list with the relevant article's ids."""
        return list(set(self.invertedIndex.get(term.lower(), EMPTY_SET).keys()))

    def searchAnd(self, term1, term2):
        """Searches for documents containing both term1 and term2
        and returns a list with the relevant article's ids."""
        return list(
            set(self.invertedIndex.get(term1.lower(), EMPTY_SET).keys()) & 
            set(self.invertedIndex.get(term2.lower(), EMPTY_SET).keys())
        )

    def searchOr(self, query):
        """Searches for documents containing term1 or term2
        and returns a list with the relevant article's ids."""
        result = list(set(self.tfIdfSearch(query)))
        return result[:5] if (len(result) > 5) else result

    
    def getArticles(self, idList):
        articles = []
        for id in idList:
            articles.append(self.articles[id])
        return articles
    
    