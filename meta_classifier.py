#!/usr/bin/env python3

import os
import sys

import joblib
import argparse
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


def create_confusion_matrix(confusion_m, categories, y_lim_value=5, title="cm", show_plots=False, method="TRAINING"):
    '''
    Creates a confusion matrix
    '''
    plt.figure(figsize = (16, 9), dpi=150)
    sn.set(font_scale=2.5) #label size
    hm = sn.heatmap(confusion_m, annot=True, fmt='g', annot_kws={"size": 32}) #font size
    hm.set_ylim(y_lim_value, 0)
    hm.set(xticklabels = categories, yticklabels = categories)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    plt.title(title + ' Confusion Matrix')
    if show_plots:
        plt.show()
    hm.figure.savefig(method + "_" + title + '_confusion_matrix' + '.png', figsize = (16, 9), dpi=150, bbox_inches="tight")
    plt.close()


def train(pipeline, X, y, categories, show_plots=False, show_cm=False, show_report=False, folds=10, title="title"):
    '''
    Train the classifier and evaluate the results
    '''
    print(f"\n#### TRAINING... [{title}]")
    X = np.array(X)
    y = np.array(y)
    y_bias = np.array([bias for hyperp, bias in y])
    inverse_dict = ["right", "left-center", "left", "least", "right-center", "true", "false"]
    hyperp_cat = ["false", "true"]
    
    try:
        print(f"Classifier used: {pipeline.named_steps['clf']}")
    except AttributeError as e:
        print(f"Using Stacking Classifier")

    if title=="StackingClassifier":
        show_cm = True
        show_report = True

    accuracy_bias, accuracy_hyperp = 0, 0
    confusion_m_bias = np.zeros(shape=(len(categories),len(categories)))
    confusion_m_hyperp = np.zeros(shape=(len(hyperp_cat),len(hyperp_cat)))
    kf = StratifiedKFold(n_splits=folds).split(X, y_bias)
    pred_overall_bias = np.array([])
    y_test_overall_bias = np.array([])
    pred_overall_hyperp = np.array([])
    y_test_overall_hyperp = np.array([])
    for train_index, test_index in kf:
        y_hyperp = np.array([hyperp for hyperp, bias in y])    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_bias[train_index], y_bias[test_index]
        y_train_hyperp, y_test_hyperp = y_hyperp[train_index], [inverse_dict[hp] for hp in y_hyperp[test_index]]
        trained  = pipeline.fit(X_train, y_train) 
        pred_bias = pipeline.predict(X_test)
        pred_hyperp = ["true" if (inverse_dict[bias]=="left" or inverse_dict[bias]=="right") else "false" for bias in pred_bias]
        accuracy_bias += accuracy_score(y_test, pred_bias)
        accuracy_hyperp += accuracy_score(y_test_hyperp, pred_hyperp)
        confusion_m_bias = np.add(confusion_m_bias, confusion_matrix(y_test, pred_bias, labels=categories))
        confusion_m_hyperp = np.add(confusion_m_hyperp, confusion_matrix(y_test_hyperp, pred_hyperp, labels=hyperp_cat))

        pred_overall_bias = np.concatenate([pred_overall_bias, pred_bias])
        y_test_overall_bias = np.concatenate([y_test_overall_bias, y_test])

        pred_overall_hyperp = np.concatenate([pred_overall_hyperp, pred_hyperp])
        y_test_overall_hyperp = np.concatenate([y_test_overall_hyperp, y_test_hyperp])

    print("\nAverage accuracy (bias): %.6f"%(accuracy_bias/folds) + "\n")
    print("Average accuracy (hyperpartisan): %.6f"%(accuracy_hyperp/folds) + "\n")

    if show_report:
        print('Classification report [bias]\n')
        print(classification_report(y_test_overall_bias, pred_overall_bias, digits=2))
        print('Classification report [hyperpartisan]\n')
        print(classification_report(y_test_overall_hyperp, pred_overall_hyperp, digits=2))
    if show_cm:        
        print('\nConfusion matrix (bias)\n')
        print(confusion_m_bias)
        print('Confusion matrix (hyperpartisan)\n')
        print(confusion_m_hyperp)

    create_confusion_matrix(confusion_m_bias, inverse_dict[:5], y_lim_value=5.0, title=title+"_bias_", show_plots= show_plots)
    create_confusion_matrix(confusion_m_hyperp, hyperp_cat, y_lim_value=2.0, title=title+"_hyperp_", show_plots= show_plots)          


def test(classifier, Xtest, Ytest, show_cm=False, show_plots=False, show_report=False, title="title"):
    '''
    Test the classifier and evaluate the results
    '''    
    inverse_dict = ["right", "left-center", "left", "least", "right-center", "true", "false"]
    joint_labels = ["false least", "false left-center", "false right-center", "true left", "true right"]
    Yguess = classifier.predict(Xtest)
    Ytest_bias = [inverse_dict[bias] for hyperp, bias in Ytest]
    Yguess_bias = [inverse_dict[bias] for bias in Yguess]
    Ytest_hyperp = [inverse_dict[hyperp] for hyperp, bias in Ytest]
    Yguess_hyperp = ["true" if (inverse_dict[bias]=="left" or inverse_dict[bias]=="right") else "false" for bias in Yguess]
    Ytest_joint = [" ".join((inverse_dict[hyperp], inverse_dict[bias])) for hyperp, bias in Ytest]
    Yguess_joint = [" ".join(("true", inverse_dict[bias])) if (inverse_dict[bias]=="left" or inverse_dict[bias]=="right") \
        else " ".join(("false", inverse_dict[bias])) for bias in Yguess]

    print(f"\n#### TESTING... [{title}]")
    try:
        print(f"Classifier used: {classifier.named_steps['clf']}")
    except AttributeError as e:
        print(f"Using Stacking Classifier")

    if title=="StackingClassifier":
        show_cm = True
        show_report = True        
        with open('output_stacked_clf.txt', 'w') as f:
            for x_id, bias, hyperp in zip([i[0] for i in Xtest], Yguess_bias, Yguess_hyperp):
                f.write(str(x_id) + " " + hyperp + " " + bias + "\n")

    confusion_m_bias = np.zeros(shape=(len(inverse_dict[:5]), len(inverse_dict[:5])))
    confusion_m_hyperp = np.zeros(shape=(len(inverse_dict[5:]), len(inverse_dict[5:])))
    confusion_m_joint = np.zeros(shape=(len(joint_labels), len(joint_labels)))

    print(f"\naccuracy (bias) = {round(accuracy_score(Ytest_bias, Yguess_bias), 5)}")
    print(f"accuracy (hyperp) = {round(accuracy_score(Ytest_hyperp, Yguess_hyperp), 5)}")
    print(f"accuracy (joint) = {round(accuracy_score(Ytest_joint, Yguess_joint), 5)}\n")

    if show_report:
        print('Classification report [bias]\n')
        print(classification_report(Ytest_bias, Yguess_bias))
        print('Classification report [hyperpartisan]\n')
        print(classification_report(Ytest_hyperp, Yguess_hyperp))
        print('Classification report [joint]\n')
        print(classification_report(Ytest_joint, Yguess_joint))

    confusion_m_bias = np.add(confusion_m_bias, confusion_matrix(Ytest_bias, Yguess_bias, labels = inverse_dict[:5]))
    confusion_m_hyperp = np.add(confusion_m_hyperp, confusion_matrix(Ytest_hyperp, Yguess_hyperp, labels = inverse_dict[5:]))
    confusion_m_joint = np.add(confusion_m_joint, confusion_matrix(Ytest_joint, Yguess_joint, labels = joint_labels))
    if show_cm:
        print('\nConfusion matrix [bias]')
        print(confusion_m_bias)
        print('\nConfusion matrix [hyperpartisan]')
        print(confusion_m_hyperp)
        print('\nConfusion matrix [joint]')
        print(confusion_m_joint)

    create_confusion_matrix(confusion_m_bias, inverse_dict[:5], y_lim_value=5.0, title=title+"_bias_", show_plots=show_plots, method="TESTING")
    create_confusion_matrix(confusion_m_hyperp, inverse_dict[5:], y_lim_value=2.0, title=title+"_hyperp", show_plots=show_plots, method="TESTING")
    create_confusion_matrix(confusion_m_joint, joint_labels, y_lim_value=5.0, title=title+"_joint", show_plots=show_plots, method="TESTING")


def main(argv):
    parser = argparse.ArgumentParser(description='Control everything')
    parser.add_argument('files', nargs="+")
    parser.add_argument('--model', help="Please provide a .pkl model")
    parser.add_argument('--save', help="Use: --save [filename] ; Saves the model, with the given filename")
    args = parser.parse_args()

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

    if len(args.files) == 1:
        X, Y = read_and_process(args.files[0], train=True)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
        # Xtrain, Ytrain = X, Y # to train on all training data
    elif len(args.files) == 2:
        Xtrain, Ytrain = read_and_process(args.files[0], title="Train", train=True)
        Xtest, Ytest = read_and_process(args.files[1], title="Test")
    else:
        print("Usage: python3 meta_classifier.py <trainset> <testset>", file=sys.stderr)

    translation_dict = {"right": 0, "left-center": 1, "left": 2, "least": 3, "right-center": 4, "true": 5, "false": 6}
    Ytrain = np.array([(translation_dict[hyperp], translation_dict[bias]) for hyperp, bias in Ytrain])
    Ytest = np.array([(translation_dict[hyperp], translation_dict[bias]) for hyperp, bias in Ytest])

    categories = [0, 1, 2, 3, 4] 

    classifier_words = model_words()
    classifier_title = model_title()
    classifier_meta = model_meta()

    do_grid_search(Xtrain, Ytrain, classifier_words, parameters[kernel], title=kernel, start = False)
    do_grid_search(Xtrain, Ytrain, classifier_title, parameters[kernel], title=kernel, start = False)

    sclf = StackingCVClassifier(classifiers=[classifier_title, classifier_words],
                                use_probas=False,
                                meta_classifier=classifier_meta,
                                random_state=42)                      

    if args.model:
        the_classifier = joblib.load(args.model)
        test(the_classifier, Xtest, Ytest, title='StackingClassifier')
    else:
        for clf, label in zip([classifier_title, classifier_words, sclf], ['Title_SVM', 'Words_SVM', 'StackingClassifier']):
            train(clf, Xtrain, Ytrain, categories, show_report=False, title=label, folds=5)
            if args.save:
                joblib.dump(clf, args.save+".pkl") 
            test(clf, Xtest, Ytest, title=label)

if __name__ == '__main__':
    main(sys.argv)

