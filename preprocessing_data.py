#!/usr/bin/env python3

import collections
import re
import sys
from collections import Counter

import pandas as pd
import progressbar
import spacy
import tensorflow_datasets as tfds
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction import stop_words
from sklearn.preprocessing import LabelBinarizer

nlp = spacy.load("en_core_web_sm")
snow = SnowballStemmer('english')

def uniques(words, badwords=False):
    '''
    Returns set of unique words, and filters badwords from them
    '''
    if not badwords:
        return set(words)
    return set(words) - set(badwords)


def read_corpus(corpus_file):
    '''
    Create a bag-of-words and labels from a file
    '''
    tokenizer = tfds.features.text.Tokenizer()
    nltk_stopword_set = set(stopwords.words('english')) #179 words
    scikit_stopword_set = set(stop_words.ENGLISH_STOP_WORDS) #318 words
    union_stopword_set = nltk_stopword_set | scikit_stopword_set # 378 words    
    documents, labels = [], []
    tokenizer = tfds.features.text.Tokenizer() 
    data = pd.read_csv(corpus_file, sep='\t')

    print("\n ### Distribution of the data")
    print(data.groupby("bias").size(), end="\n\n")

    for item in data.itertuples():
        text = item.text
        title = item.title
        if type(item.text) != str:
            text = ""
        if type(item.title) != str:
            title = ""
        documents.append((tokenizer.tokenize(title.strip()),
                         tokenizer.tokenize(text.strip())))
        labels.append(item.bias)
        
    return documents, labels


def preprocessing(documents):
    '''
    Some simple pre-processing, for example changing all numbers to "number"
    '''
    for i in range(len(documents)):
        documents[i] = [re.sub(r'[0-9]+','number', doc) for doc in documents[i]]
        # documents[i] = [re.sub(r'<[^<>]+>','', doc) for doc in documents[i]]
        # documents[i] = [re.sub(r'(http|https): //[^\s]*','httpaddr', doc) for doc in documents[i]]
        documents[i] = [re.sub(r'[^\s]+@[^\s]+','emailaddr', doc) for doc in documents[i]]
        # documents[i] = [re.sub(r'[$|€]+','money', doc) for doc in documents[i]]
        documents[i] = [re.sub(r'[^a-zA-Z0-9]', ' ', doc) for doc in documents[i]]
        documents[i] = [doc.strip() for doc in documents[i] if doc != " " and len(doc)>1]


def get_high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
    '''
    Gets the high information words using chi square measure 
    '''
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    
    for label, words in labelled_words:
        for word in words:
            word_fd[word] += 1
            label_word_fd[label][word] += 1
    
    n_xx = label_word_fd.N()
    high_info_words = set()
    
    for label in label_word_fd.conditions():
        n_xi = label_word_fd[label].N()
        word_scores = collections.defaultdict(int)
        
        for word, n_ii in label_word_fd[label].items():
            n_ix = word_fd[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score
        
        bestwords = [word for word, score in word_scores.items() if score >= min_score]
        high_info_words |= set(bestwords)
    
    return high_info_words


def high_information_words(X, y, title):
    '''
    Get and display info on high info words
    '''
    print(f"\n#### OBTAINING HIGH INFO WORDS [{title}]...")

    labelled_words = []
    amount_words = 0
    distinct_words = set()
    for words, genre in zip(X, y):
        labelled_words.append((genre, words))
        amount_words += len(words)
        for word in words:
            distinct_words.add(word)

    high_info_words = set(get_high_information_words(labelled_words, BigramAssocMeasures.chi_sq, 7)) # 7

    print("\tNumber of words in the data: %i" % amount_words)
    print("\tNumber of distinct words in the data: %i" % len(distinct_words))
    print("\tNumber of distinct 'high-information' words in the data: %i" % len(high_info_words))

    return high_info_words


def return_high_info(X, y, title="data"):
    '''
    Return list of high information words per document
    '''
    try:
        high_info_words = high_information_words(X, y, title)

        X_high_info = []
        for bag in X:
            new_bag = []
            for words in bag:
                if words in high_info_words:
                    new_bag.append(words)
            X_high_info.append(new_bag)
    except ZeroDivisionError:
        print("Not enough information too get high information words, please try again with more files.", file=sys.stderr)
        X_high_info = X
    return X_high_info


def return_named_ent(X, title="data"):
    '''
    Return list of named entities per document
    '''
    print(f"\n#### RETRIEVING NAMED ENTITIES TAGS [{title}]...")    
    named_ent = []
    for bag, _ in zip(X, progressbar.progressbar(range(len(X)))):
        new_bag = []
        for ent in nlp(" ".join(bag)).ents:
            new_bag.append(ent.label_)
        named_ent.append(new_bag)
    return named_ent
    

def return_pos_tagged(X, title="data"):
    '''
    Return list of part-of-speech tags per document
    '''
    print(f"\n#### RETRIEVING PART-OF-SPEECH TAGS [{title}]...")    
    pos_tag = []
    for bag, _ in zip(X, progressbar.progressbar(range(len(X)))):
        new_bag = []
        for pos in nlp(" ".join(bag)):
            new_bag.append(pos.tag_)
        pos_tag.append(new_bag)
    print()
    return named_ent


def read_and_process(file):
    '''
    Reads in data from file to pandas dataframe, and preprocesses the data for the model
    '''
    X, Y = read_corpus(file)
    
    categories = set(Y)
    
    preprocessing([text for title, text in X])

    X_high_info = return_high_info([text for title, text in X], Y, "data")
    
    # X_pos = return_pos_tagged(X, "data")
    
    # X_ent = return_named_ent(X, "data")
    
    X = [(title, words, x_high) for (title, words), x_high in zip(X, X_high_info)]
    encoder = LabelBinarizer()
    Y = encoder.fit_transform(Y) 

    return X, Y
