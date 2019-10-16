#!/usr/bin/env python3

import os
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import collections
import re
import sys
import tensorflow as tf
from collections import Counter

import pandas as pd
import numpy as np
import progressbar
import seaborn as sn
import tensorflow_datasets as tfds
import spacy
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.tree import DecisionTreeClassifier

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

    for item in data.itertuples():
        text = item.text
        title = item.title
        if type(item.text) != str:
            text = ""
        if type(item.title) != str:
            title = ""
        documents.append(tokenizer.tokenize(title.strip()),
                         tokenizer.tokenize(text.strip()))
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


def identity(x):
    return x


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


# def return_named_ent(X, title="data"):
#     '''
#     Return list of named entities per document
#     '''
#     print(f"\n#### RETRIEVING NAMED ENTITIES TAGS [{title}]...")    
#     named_ent = []
#     for bag, _ in zip(X, progressbar.progressbar(range(len(X)))):
#         new_bag = []
#         for ent in nlp(" ".join(bag)).ents:
#             new_bag.append(ent.label_)
#         named_ent.append(new_bag)
#     return named_ent
    

def return_pos_tagged(X, title="data"):
    '''
    Return list of part-of-speech tags per document
    '''
    print(f"\n#### RETRIEVING PART-OF-SPEECH TAGS [{title}]...")    
    named_ent = []
    for bag, _ in zip(X, progressbar.progressbar(range(len(X)))):
        new_bag = []
        for ent in nlp(" ".join(bag)):
            new_bag.append(ent.tag_)
        named_ent.append(new_bag)
    print()
    return named_ent


def train(pipeline, X, y, categories, show_plots=False):
    '''
    Train the classifier and evaluate the results
    '''
    print("\n#### TRAINING...")

    X = np.array(X)
    y = np.array(y)
    
    print(f"Classifier used: {pipeline.named_steps['clf']}")

    accuracy = 0
    confusion_m = np.zeros(shape=(len(categories),len(categories)))
    kf = StratifiedKFold(n_splits=10).split(X, y)
    pred_overall = np.array([])
    y_test_overall = np.array([])
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        trained  = pipeline.fit(X_train, y_train) 
        pred = pipeline.predict(X_test)
        accuracy += accuracy_score(y_test, pred)
        confusion_m = np.add(confusion_m, confusion_matrix(y_test, pred, labels=categories))
        
        pred_overall = np.concatenate([pred_overall, pred])
        y_test_overall = np.concatenate([y_test_overall, y_test])

    print("Average accuracy: %.6f"%(accuracy/10) + "\n")

    print (classification_report(y_test_overall, pred_overall, digits=3))
    print('Confusion matrix')
    print(confusion_m)

    plt.figure(figsize = (16, 9), dpi=150)
    sn.set(font_scale=1.4) #label size
    hm = sn.heatmap(confusion_m, annot=True, fmt='g', annot_kws={"size": 16}) #font size
    hm.set(xticklabels = categories, yticklabels = categories)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    plt.title(str(pipeline.named_steps['clf']).split("(")[0] + ' Confusion Matrix')
    if show_plots:
        plt.show()

    hm.figure.savefig('TRAINING_'+str(pipeline.named_steps['clf']).split("(")[0] + '_confusion_matrix' + '.png', figsize = (16, 9), dpi=150)

    plt.close()


def do_grid_search(X, y, pipeline, parameters, title="", start=False):
    '''
    Do 5 fold cross-validated gridsearch over certain parameters and
    print the best parameters found according to accuracy
    '''
    if not start:
        print("\n#### SKIPPING GRIDSEARCH...")
    else:
        print(f"\n#### GRIDSEARCH [{title}] ...")
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5, scoring='accuracy', return_train_score=True) 
        grid_search.fit(X, y)

        df = pd.DataFrame(grid_search.cv_results_)[['params','mean_train_score','mean_test_score']]
        print(f"\n{df}\n")

        # store results for further evaluation
        with open('grid_' + title + '_pd.pickle', 'wb') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)        
   
        print("Best score: {0}".format(grid_search.best_score_))  
        print("Best parameters set:")  
        best_parameters = grid_search.best_estimator_.get_params()  
        for param_name in sorted(list(parameters.keys())):  
            print("\t{0}: {1}".format(param_name, best_parameters[param_name]))       


class FeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, subs):
        features = {}
        features['text'] = [item[0] for item in subs]
        # features['char'] = [" ".join(item[0]) for item in subs]
        features['text_high'] = [item[1] for item in subs]
        features['pos_tag'] = [item[2] for item in subs]
        # features['named_ent'] = [item[3] for item in subs]

        return features


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


def main(argv):
    show_plots = False
    if len(argv) == 2:
        X, Y = read_corpus(argv[1])
        split_point = int(0.75*len(X))
        Xtrain = X[:split_point]
        Ytrain = Y[:split_point]
        Xtest = X[split_point:]
        Ytest = Y[split_point:]
    elif len(argv) == 3:
        Xtrain, Ytrain = read_corpus(argv[1])
        Xtest, Ytest = read_corpus(argv[2])
    else:
        print("Usage: python3 LFDassignment2.py <trainset> <testset>", file=sys.stderr)

    return 0

    categories = ['books', 'camera', 'dvd', 'health', 'music', 'software']
    
    preprocessing(Xtrain)
    preprocessing(Xtest)

    X_high_info_train = return_high_info(Xtrain, Ytrain, "train_data")
    X_high_info_test = return_high_info(Xtest, Ytest, "test_data")

    X_pos_train = return_pos_tagged(Xtrain, "train_data")
    X_pos_test = return_pos_tagged(Xtest, "test_data")

    # X_ent_train = return_named_ent(Xtrain, "train_data")
    # X_ent_test = return_named_ent(Xtest, "test_data")

    Xtrain = [(x, x_high, x_pos) for x, x_high, x_pos in zip(Xtrain, X_high_info_train, X_pos_train)]
    Xtest = [(x, x_high, x_pos) for x, x_high, x_pos in zip(Xtest, X_high_info_test, X_pos_test)]

    clfs = [svm.SVC(kernel='linear', C=1.0)]
    
    parameters = {
        'linear': {  
            'clf__C': np.logspace(-3, 2, 6),
        },
        'rbf': {
            'clf__C': np.logspace(-3, 2, 6),
            'clf__gamma': np.logspace(-3, 2, 6),
            'clf__kernel': ['rbf']
        }
    }
    kernel = 'linear'

    classifier = Pipeline([
        # Extract the features
        ('features', FeaturesExtractor()),
        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            # n_jobs = -1,
            transformer_list = [
                # Pipeline bag-of-words model 
                ('words', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                                              max_df = .2, ngram_range = (1,2))),
                    ('chi-square', SelectKBest(chi2, 3000)),
                ])),

                # # Pipeline for character features
                # ('chars', Pipeline([
                #     ('selector', ItemSelector(key='char')),
                #     ('tfidf', TfidfVectorizer(analyzer='char', preprocessor = identity, tokenizer = identity, ngram_range=(3,10))),
                # ])),

                # Pipeline for high info words bag-of-words model 
                ('text_high', Pipeline([
                    ('selector', ItemSelector(key='text_high')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                                              max_df = .2)),
                ])),

                # Pipeline for POS tags
                ('pos_tag', Pipeline([
                    ('selector', ItemSelector(key='pos_tag')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity)),
                ])),

                # Pipeline for named entity tags
                # ('named_ent', Pipeline([
                #     ('selector', ItemSelector(key='named_ent')),
                #     ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity)),
                # ])),

            ],

            # weight components in FeatureUnion
            transformer_weights={ 
                # 'text': .3,
                # 'chars': .4,
                # 'text_high' : .7,
                # 'pos_tag': .1,
            },
        )),
        # Use a classifier on the combined features
        ('clf', clfs[2]),
    ])

    do_grid_search(Xtrain, Ytrain, classifier, parameters[kernel], title=kernel, start = False)

    train(classifier, Xtrain, Ytrain, categories, show_plots = show_plots)
    Yguess = classifier.predict(Xtest)

    print("\n#### TESTING...")
    print(f"Classifier used: {classifier.named_steps['clf']}")

    confusion_m = np.zeros(shape=(len(categories),len(categories)))

    print(f"accuracy = {round(accuracy_score(Ytest, Yguess), 5)}\n")

    print(classification_report(Ytest, Yguess))
    confusion_m = np.add(confusion_m, confusion_matrix(Ytest, Yguess, labels = categories))
    print('Confusion matrix')
    print(confusion_m)

    plt.figure(figsize = (10, 5), dpi = 150)
    sn.set(font_scale = 1.4) 
    hm = sn.heatmap(confusion_m, annot = True, fmt = 'g', annot_kws = {"size": 16}) 
    hm.set(xticklabels = categories, yticklabels = categories)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    plt.title('Confusion Matrix ' + type(classifier['clf']).__name__)
    if show_plots:
        plt.show()
    plt.savefig('TESTING_' + type(classifier['clf']).__name__ + "_confusion_matrix.png")

if __name__ == '__main__':
    main(sys.argv)
