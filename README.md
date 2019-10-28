# LFD_news_detection
Hyper Partisan News Detection (SemEval019)

## Requirements

 - Python3
 - Python packages in `requirements.txt`

## Usage

### RNN Model (primary model)

 - Download data files the following URL to use the pretrained model.
 - Put the downloaded `rnn-model.pt` and `hyperp-training-grouped.csv` in `data/`
 - Run `python3 rnn.py data/hyperp-training-grouped.csv TESTFILE.csv.xz data/rnn-model.pt`

