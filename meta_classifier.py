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

from preprocessing_data import read_and_process
from ml_models import model_words, model_title

from sklearn.model_selection import train_test_split

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

    print(classification_report(y_test_overall, pred_overall, digits=3))
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


def main(argv):

    ################# REMOVE LATER #######################

    print("#### Getting Pickle File")

    import pickle    

    with open('small_data.pickle', 'rb') as handle:
        Xtrain, Xtest, Ytrain, Ytest = pickle.load(handle)

    ######################################################

    # if len(argv) == 2:
    #     X, Y = read_and_process(argv[1])
    #     Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=42)
    # elif len(argv) == 3:
    #     Xtrain, Ytrain = read_and_process(argv[1])
    #     Xtest, Ytest = read_and_process(argv[2])
    # else:
    #     print("Usage: python3 meta_classifier.py <trainset> <testset>", file=sys.stderr)

    # with open('data.pickle', 'wb') as handle:
    #     pickle.dump((Xtrain, Xtest, Ytrain, Ytest), handle, protocol=pickle.HIGHEST_PROTOCOL)

    categories = list(set(Ytrain+Ytest))

    classifier_words = model_words()
    classifier_title = model_title()

    train(classifier_words, Xtrain, Ytrain, categories)

if __name__ == '__main__':
    main(sys.argv)