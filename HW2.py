#Aria Kim & Nambi Williams
#HW2

from collections import Counter, namedtuple
from math import log
import re
import nltk
import sys 
reload(sys) 
sys.setdefaultencoding('utf8')

LanguageModel = namedtuple('LanguageModel', 'num_tokens, vocab, nminus1grams, unknowns, ngrams') # holds counts for the lm
DELIM = "_" # delimiter for tokens in an ngram

def tokenize_text(text, tag):
    """ Converts a string to a list of tokens """
    tokens = []
    #Total number of sentences given the tag plot or review
    for sent in nltk.sent_tokenize(text):
        if tag in sent:
            #Removes tag from text
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
    #Number of unknown tokens
    unknowns = 0
    #Considers tokens with count below threshold 2 as unknown
    if n is 1: #unigram
        for token in ngrams:
            if ngrams.get(token) is 1:
                unknowns += 1
    else: #bigram           
        for token in nminus1grams:
            if nminus1grams.get(token) is 1:
                unknowns += 1
                
    return LanguageModel(num_tokens, vocab, nminus1grams, unknowns, ngrams)


def discount(lm, token, history):
    d = 0.5
    #Ngram and history counts
    ngram_count_l = lm.ngrams.get(token, 0) - d
    prefix_count_l = lm.nminus1grams.get(history, 0)
    return abs(float(ngram_count_l) / prefix_count_l)

def alpha(lm, history):
    #Initial counts of seen and unseen probabilities
    seen = 0
    unseen = 0
    for toke in lm.vocab: #
        p = history + "_" +toke
        #Absolute discounting for seen and unseen ngrams
        if p in lm.ngrams:
            seen += discount(lm, p, history)
        else: 
            unseen += float(lm.nminus1grams.get(toke))/lm.num_tokens

    return float(1 - seen)/unseen

def katzProb(lm, token, history, n):
    #Calculates Katz Probability for known and unknown ngrams
    if n is 1:  #unigram case
        if token not in lm.ngrams:
            return log(float(lm.unknowns)/lm.num_tokens)
        else:
            return log(float(lm.ngrams.get(token))/lm.num_tokens)
    
    if history + "_" + token in lm.ngrams: #bigram
        return discount(lm, token, history)
    else:
        if token not in lm.nminus1grams: 
            return log(alpha(lm,history) * (float(lm.unknowns)/lm.num_tokens))
        else:
            return log(alpha(lm, history)* (float(lm.nminus1grams.get(token))/lm.num_tokens))

def classify(lmplot, lmreview, test, n):
    #Vars to check accuracy
    isplot = True
    correct = 0
    incorrect = 0
    
    for sent in nltk.sent_tokenize(test):
       #When sentence is tagged as plot
       if(sent[:3] is 'p: '):
           sent = sent.replace('p: ','')
       #When sentence is tagged as review
       else:
           sent = sent.replace('r: ','')
           #It's not a plot sentence
           isplot = False
           
       #Initial plot and review probabilities 
       plot = 1.0
       rev = 1.0
       #Initializes history
       history = ""
       
       #Calculates P(token|c)
       for token in sent:
           if history is not "":
               plot = plot * katzProb(lmplot, token, history, n)
               rev = rev * katzProb(lmreview, token, history, n)
           history = token
           
       #print plot, rev
       #Classification of sentence
       #Checks accuracy
       if plot > rev:
           if isplot:
               correct += 1
           else:
               incorrect += 1
           print 'p: ' + sent
       else:
           if isplot:
               incorrect += 1
           else:
               correct += 1
           print 'r: ' + sent
           
    return float(correct)/(correct + incorrect)


if __name__ == '__main__':
    """ example usage"""
    #Language Model from training text
    with open("hw2_train.txt", 'r') as file:
        text = file.read()

    #biplot = build_lm(text,'p: ', 2) #bigram model
    #bireview = build_lm(text,'r: ',2)

    #testing 2.1b, 2.1c
    #print(katzProb(biplot, "is", "there", 2))
    #print(katzProb(bireview, "is", "there", 2))     #add more testing

    #Testing text
    with open("hw2_test.txt", 'r') as file:
        test = file.read()
        
    #2.1d
    #print classify(biplot, bireview, test, 2)

    #2.1e
    hetplot = build_lm(text,'p: ', 1)
    hetreview = build_lm(text,'r: ',1)
         
    print classify(hetplot, hetreview, test, 1)
