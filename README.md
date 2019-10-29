# LFD_news_detection
Hyper Partisan News Detection (SemEval019)

## Requirements

 - Python3
 - Python packages in `requirements.txt`

## Usage

### RNN Model (primary model)

 - Download data files the following URL to use the pretrained model: http://bit.ly/2JwFxAF
 - Make sure `rnn-model.pt` and `hyperp-training-grouped.processed.csv` are in `data/`
 - Run `python3 rnn.py data/hyperp-training-grouped.processed.csv TESTFILE.csv data/rnn-model.pt`
