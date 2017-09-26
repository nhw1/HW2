#Aria Kim & Nambi Williams
#HW2

from collections import Counter, namedtuple
from math import log
import re
import nltk
import sys 
reload(sys) 
sys.setdefaultencoding('utf8')

LanguageModel = namedtuple('LanguageModel', 'num_tokens, vocab, sentcount, unknowns, ngrams') # holds counts for the lm
DELIM = "_" # delimiter for tokens in an ngram

def tokenize_text(text, tag):
    """ Converts a string to a list of tokens """
    tokens = []
    #Total number of sentences given the tag plot or review
    sentcount = 0
    for sent in nltk.sent_tokenize(text):
        if tag in sent:
            #Removes tag from text, and adds count to total
            sent = sent.replace(tag, '')
            tokens.extend(nltk.word_tokenize(sent))
            sentcount += 1
            
    print sentcount
    return tokens, sentcount

def generate_ngrams(tokens):
    """ Returns a list of ngrams made from a list of tokens """
    ngrams = []
    for i in range(0, len(tokens)):
        ngrams.append(DELIM.join(tokens[i:i+1]))
 
    return ngrams

def build_lm(text, tag):
    """ Builds an ngram language model. """
    tokentuple = tokenize_text(text, tag)
    num_tokens = len(tokentuple[0])
    vocab = set(tokentuple[0])
    ngrams = Counter(generate_ngrams(tokentuple[0]))
    #Number of unknown tokens
    unknowns = 0
    #Number of sentences for a given tag
    sentcount = tokentuple[1]
    #Considers tokens with count below threshold 2 as unknown
    for token in ngrams:
        if ngrams.get(token) is 1:
            unknowns += 1
                
    return LanguageModel(num_tokens, vocab, sentcount, unknowns, ngrams)

def bayesProb(lm, token):

    #Returns log probability for unseen unigram
    if token not in lm.ngrams:
        return log(float(lm.unknowns)/lm.num_tokens)
    else:
    #Returns Bayes probability for known unigrams
        return log(float(lm.ngrams.get(token))/lm.num_tokens)

def classify(lmplot, lmreview, test):
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
       #Calculates P(token|c)
       for token in sent:
           plot = plot * bayesProb(lmplot, token)
           rev = rev * bayesProb(lmreview, token)
       print plot, rev
       #Calculates probability with P(c)
       plot = plot * float(lmplot.sentcount)/(lmplot.sentcount + lmreview.sentcount)
       rev = rev * float(lmreview.sentcount)/(lmplot.sentcount + lmreview.sentcount)

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
           
    #Accuracy percentage
    return float(correct)/(correct + incorrect)


if __name__ == '__main__':
    """ example usage"""
    #Language Model from training text
    with open("hw2_train.txt", 'r') as file:
        text = file.read()

    plot = build_lm(text, 'p: ')
    review = build_lm(text, 'r: ')

    #Testing text
    with open("hw2_test.txt", 'r') as file:
        test = file.read()

    print classify(plot, review, test)
