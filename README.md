# LFD_news_detection
Hyper Partisan News Detection (SemEval019)

## Requirements

 - Python3
 - Python packages in `environment.yml` (using Anaconda)

## Usage

### RNN Model (primary model)

 - Download data files the following URL to use the pretrained model: http://bit.ly/2JwFxAF
 - Make sure `rnn-model.pt` and `hyperp-training-grouped.processed.csv` are in `data/`
 - Run `python3 rnn.py data/hyperp-training-grouped.processed.csv TESTFILE.csv data/rnn-model.pt`

### Stacked SVM
 
 To run the model from scratch, use the following command:
 - `python3 meta_classifier TRAININGFILE.csv TESTFILE.csv`

 To test using the pre-trained model, use the following command (this will only show test results):
 - `python3 meta_classifier TRAININGFILE.csv TESTFILE.csv --model model_stacked_svm.pkl` 
