from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn import svm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer

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


def identity(x):
    return x


def model_words():
    clfs = [svm.SVC(kernel='linear', C=1.0)]
    
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
        ('clf', clfs[0]),
    ])
    return classifier


def model_title():
    clfs = [svm.SVC(kernel='linear', C=1.0)]
    
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
        ('clf', clfs[0]),
    ])
    return classifier    