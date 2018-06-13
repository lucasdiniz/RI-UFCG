import pandas as pd

import numpy as np
from scipy import sparse
import nltk
from nltk import bigrams    
import scipy.sparse as sps

def co_occurrence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    n = len(vocab)
   
    vocab_to_index = {word:i for i, word in enumerate(vocab)}
    
    bi_grams = list(bigrams(corpus))

    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))

    I=list()
    J=list()
    V=list()
    
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]

        I.append(vocab_to_index[previous])
        J.append(vocab_to_index[current])
        V.append(count)
        
    co_occurrence_matrix = sparse.coo_matrix((V,(I,J)), shape=(n,n))

    return co_occurrence_matrix, vocab_to_index