from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn import svm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier 

class FeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, subs):
        features = {}
        features['title'] = [title for (title, words, x_high) in subs]
        # features['char'] = [" ".join(item[1]) for item in subs]
        features['text'] = [words for (title, words, x_high) in subs]
        features['text_high'] = [x_high for (title, words, x_high) in subs]
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
            #n_jobs = -1,
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


def model_test():

    # SCORES ON WORDS [TRAINING | TEST | STACKING TRAINING | STACKING TEST] (WITHOUT ANY TUNING ON SMALL_DATA with MORE BALANCED)
    # KNN =                     KUT  | KUT  | KUT  | KUT  
    # RandomForest =            0.74 | 0.72 | 0.65 | 0.72
    # MultinomialNB =           0.67 | 0.66 | 0.56 | 0.66
    # LogisticRegression =      0.73 | 0.72 | 0.68 | 0.72 
    # LogisitcRegressionCV =    0.79 | 0.76 | 0.70 | 0.76   (SLOW)
    # '' '' + solver=saga =     0.78 | 0.75 | SLOW | SLOW   (SLOW*2)
    # NuSVC =                   0.52 | 0.62 | 0.41 | 0.42   (SLOW)
    # NuSVC (Gamme=Scale) =     0.68 | 0.72 | 0.68 | 0.72   (SLOW)
    # LinearSVC =               0.79 | 0.76 | 0.70 | 0.76 
    # PassiveAgressive =        0.79 | 0.76 | 0.70 | 0.76

    clfs = [KNeighborsClassifier(n_neighbors=5), 
            RandomForestClassifier(random_state=42), 
            MultinomialNB(), 
            LogisticRegression(),
            NuSVC(gamma="scale"),  
            LinearSVC(),
            LogisticRegressionCV(multi_class="multinomial", solver="saga"),
            PassiveAggressiveClassifier(),
            ]
    
    classifier = Pipeline([
        # Extract the features
        ('features', FeaturesExtractor()),
        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list = [
                # Pipeline bag-of-words model 
                ('words', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                                              max_df = .2)),
                    ('chi-square', SelectKBest(chi2, 3000)),
                ])),
                # Pipeline for high info words bag-of-words model 
                ('text_high', Pipeline([
                    ('selector', ItemSelector(key='text_high')),
                    ('tfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                                              max_df = .2)),
                ])),
            ],
        )),
        # Use a classifier on the combined features
        ('clf', clfs[7]),
    ])
    return classifier


def model_meta():
    '''
    The final meta classifier using the outputs from the other models as input
    '''
    return svm.SVC(kernel='linear', C=1.0)