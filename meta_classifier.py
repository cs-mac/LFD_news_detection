#!/usr/bin/env python3

import sys

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from mlxtend.classifier import StackingCVClassifier
from sklearn import model_selection
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)

from ml_models import *
from preprocessing_data import read_and_process

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


def train(pipeline, X, y, categories, show_plots=False, show_report=False, folds=10, title="title"):
    '''
    Train the classifier and evaluate the results
    '''
    print(f"\n#### TRAINING... [{title}]")
    X = np.array(X)
    y = np.array(y)

    try:
        print(f"Classifier used: {pipeline.named_steps['clf']}")
    except AttributeError as e:
        print(f"Using Stacking Classifier")

    accuracy = 0
    confusion_m = np.zeros(shape=(len(categories),len(categories)))
    kf = StratifiedKFold(n_splits=folds).split(X, y)
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

    print("\nAverage accuracy: %.6f"%(accuracy/folds) + "\n")

    if show_report:
        print(classification_report(y_test_overall, pred_overall, digits=3))
    print('Confusion matrix')
    print(confusion_m)

    plt.figure(figsize = (16, 9), dpi=150)
    sn.set(font_scale=1.4) #label size
    hm = sn.heatmap(confusion_m, annot=True, fmt='g', annot_kws={"size": 16}) #font size
    hm.set(xticklabels = categories, yticklabels = categories)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    plt.title(title + ' Confusion Matrix')
    if show_plots:
        plt.show()

    hm.figure.savefig('TRAINING_'+ title + '_confusion_matrix' + '.png', figsize = (16, 9), dpi=150)

    plt.close()
          

def test(classifier, Xtest, Ytest, categories, show_cm=False, show_plots=False, title="title"):
    Yguess = classifier.predict(Xtest)

    print(f"\n#### TESTING... [{title}]")
    try:
        print(f"Classifier used: {classifier.named_steps['clf']}")
    except AttributeError as e:
        print(f"Using Stacking Classifier")

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
    plt.title('Confusion Matrix ' + title)
    if show_plots or show_all:
        plt.show()
    plt.savefig('TESTING_' + title + "_confusion_matrix.png")


def main(argv):

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

    ################# REMOVE LATER #######################

    # print("#### Getting Pickle File")

    # import pickle    

    # with open('small_data.pickle', 'rb') as handle:
    #     Xtrain, Xtest, Ytrain, Ytest = pickle.load(handle)

    ######################################################

    if len(argv) == 2:
        X, Y = read_and_process(argv[1])
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20, random_state=42)
    elif len(argv) == 3:
        Xtrain, Ytrain = read_and_process(argv[1], title="Train", train=True)
        Xtest, Ytest = read_and_process(argv[2], title="Test")
    else:
        print("Usage: python3 meta_classifier.py <trainset> <testset>", file=sys.stderr)

    translation_dict = {"left": 0, "left-center": 1, "least": 2, "right-center": 3, "right": 4}
    Ytrain = np.array([translation_dict[i] for i in Ytrain])
    Ytest = np.array([translation_dict[i] for i in Ytest])
    
    categories = [0, 1, 2, 3, 4] 

    classifier_words = model_words()
    classifier_title = model_title()
    classifier_meta = model_meta()
    classifier_test = model_test()

    do_grid_search(Xtrain, Ytrain, classifier_words, parameters[kernel], title=kernel, start = False)
    do_grid_search(Xtrain, Ytrain, classifier_title, parameters[kernel], title=kernel, start = False)

    sclf = StackingCVClassifier(classifiers=[classifier_title, classifier_words],
                                use_probas=False,
                                meta_classifier=classifier_meta,
                                random_state=42)                      

    for clf, label in zip([classifier_title, classifier_words, sclf], ['Title_SVM', 'Words_SVM', 'StackingClassifier']):
        train(clf, Xtrain, Ytrain, categories, show_report=False, title=label, folds=5)
        test(clf, Xtest, Ytest, categories, title=label)

if __name__ == '__main__':
    main(sys.argv)
