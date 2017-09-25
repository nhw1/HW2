from collections import Counter, namedtuple
from math import log
import re
import nltk
import sys 
reload(sys) 
sys.setdefaultencoding('utf8')

LanguageModel = namedtuple('LanguageModel', 'num_tokens, vocab, nminus1grams, ngrams') # holds counts for the lm
DELIM = "_" # delimiter for tokens in an ngram

def tokenize_text(text, tag):
    """ Converts a string to a list of tokens """
    tokens = []
    for sent in nltk.sent_tokenize(text):
        if tag in sent: #label as plot
            sent = sent.replace(tag, '')
            tokens.extend(nltk.word_tokenize(sent))

    return tokens

def generate_ngrams(tokens, n):
    """ Returns a list of ngrams made from a list of tokens """
    ngrams = []
    if n > 0:
        for i in range(0, len(tokens)-n+1):
            ngrams.append(DELIM.join(tokens[i:i+n]))

    return ngrams

def build_lm(text, tag, n):
    """ Builds an ngram language model. """
    tokens = tokenize_text(text, tag)

    num_tokens = len(tokens)
    vocab = set(tokens)
    nminus1grams = Counter(generate_ngrams(tokens, n - 1))
    ngrams = Counter(generate_ngrams(tokens, n))


    return LanguageModel(num_tokens, vocab, nminus1grams, ngrams)

def prob(lm, token, history=None):

    d = 0.5
    """ Computes the probability of the ngram.

    Returns the Add-1 smoothed log-probability of 
    "token" following "history"

    Args:
        lm (LanguageModel): ngram counts from the corpus
        token: target token
        history: n-1 previous tokens or None

    Return:
        float: log-probability of the ngram
    """
    if history == None: #unigram model
        ngram_count_l = lm.ngrams.get(token, 0)
        prefix_count_l = lm.num_tokens + len(lm.vocab)
    else:
        ngram_count_l = lm.ngrams.get(history+DELIM+token, 0)
        prefix_count_l = lm.nminus1grams.get(history, 0) + len(lm.vocab)

    return float(ngram_count_l) / prefix_count_l
#pre: word has been seen
#temp: assume n > 1
def discount(lm, token, history):
    d = 0.5
    ngram_count_l = lm.ngrams.get(token, 0) - d
    prefix_count_l = lm.nminus1grams.get(history, 0)
    #print ngram_count_l, prefix_count_l
    return abs(float(ngram_count_l) / prefix_count_l)

def alpha(lm, history):
    seen = 0
    unseen = 0
    for toke in lm.vocab:
        p = history + "_" +toke
        if p in lm.ngrams:
            seen += discount(lm, p, history)
        else: #if not same history
            unseen += float(lm.nminus1grams.get(toke))/lm.num_tokens

    """for key, value in lm.ngrams.iteritems():
        #print history
        if history in key[:len(history)]:
            print key
            seen += discount(lm,key,history)
        else: #if not same history
            unseen += float(lm.ngrams.get(key))/lm.num_tokens
        #print(seen)"""
    return float(1 - seen)/unseen

def katzProb(lm, token, history):   
    if history + "_" + token in lm.ngrams:
        return discount(lm, token, history)
    else:
        #print "alpha: " , alpha(lm, history)
        #print float(1)/24
        print type(lm.nminus1grams.get(token))
        return alpha(lm, history)* (float(lm.nminus1grams.get(token))/lm.num_tokens)

def classify(lmplot, lmreview, test):
    for sent in nltk.sent_tokenize(test):
       plot = 0.0
       rev = 0.0
       history = ""
       for token in sent:
           if history is not "":
               plot += katzProb(lmplot, token, history)
               rev += katzProb(lmreview, token, history)
           history = token
       if plot > rev:
           print "p: " + sent
       else:
           print "r: " + sent


if __name__ == '__main__':
    """ example usage"""

    with open("hw2_train.txt", 'r') as file:
        text = file.read()

    lmplot = build_lm(text,'p: ', 2) #bigram model
    lmreview = build_lm(text,'r: ',2)

    history = "am"
    #probs = [(w,prob(lm, w, history)) for w in lm.vocab]
    #probs = sorted(probs, key=lambda x:x[1], reverse=True)
    #print("4 most probable words to follow '{}': {}".format(history, probs[:4]))
    print(katzProb(lmplot, "is", "there"))
    print(katzProb(lmreview, "is", "there"))

    classify(lmplot, lmreview, "hw2_test.txt")


