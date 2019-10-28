#!/usr/bin/env python3

from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC, NuSVC

class FeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, articles):
        features = {}
        features['title'] = [title for (x_id, title, words, x_high) in articles]
        # features['char'] = ["".join(words) for (x_id, title, words, x_high) in articles]
        features['text'] = [words for (x_id, title, words, x_high) in articles]
        features['text_high'] = [x_high for (x_id, title, words, x_high) in articles]
        # features['named_ent'] = [x_ent for (x_id, x_ent, title, words, x_high) in articles]

        return features


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


def identity(x):
    return x


def model_words():
    '''
    The model + pipeline for features extracted from the text
    '''
    clfs = [LinearSVC(), svm.SVC(kernel='linear', C=1.0), PassiveAggressiveClassifier()]
    
    classifier = Pipeline([
        # Extract the features
        ('features', FeaturesExtractor()),
        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            #n_jobs = -1,
            transformer_list = [
                # Pipeline bag-of-words model 
                ('words', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                                              max_df = .2)),
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
                # ('pos_tag', Pipeline([
                #     ('selector', ItemSelector(key='pos_tag')),
                #     ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity)),
                # ])),

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
    return classifier


def model_title():
    '''
    The model + pipeline for features extracted from the title
    '''    
    clfs = [LinearSVC(), svm.SVC(kernel='linear', C=1.0), PassiveAggressiveClassifier()]
    
    classifier = Pipeline([
        # Extract the features
        ('features', FeaturesExtractor()),
        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list = [
               
                # Pipeline for title words
                ('title', Pipeline([
                    ('selector', ItemSelector(key='title')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity)),
                ])),

            ],

            # weight components in FeatureUnion
            transformer_weights={ 
                # 'title': .3,
            },
        )),
        # Use a classifier on the combined features
        ('clf', clfs[2]),
    ])
    return classifier    


def model_meta():
    '''
    The final meta classifier using the outputs from the other models as input
    '''
    return svm.SVC(kernel='linear', C=1.0)
