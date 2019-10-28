import time
from torchtext.data import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from sklearn.metrics import confusion_matrix

import numpy as np

from data import get_dataset


def evaluate(model, data):
    model.eval()

    total_loss = 0.
    bias_true, bias_pred = [], []
    hyperp_true, hyperp_pred = [], []

    with torch.no_grad():
        test_iter = get_iterator(data, batch_size)
        for batch in test_iter:
            titles, title_lengths = batch.title
            texts, text_lengths = batch.text
            # publishers = batch.publisher
            bias = batch.bias
            hyperp = batch.hyperp.reshape((-1, 1)).float()

            bias_output, hyperp_output = model(titles, title_lengths, texts, text_lengths)
            total_loss += len(bias) * bias_criterion(bias_output, bias).item()
            total_loss += len(hyperp) * hyperp_criterion(hyperp_output, hyperp).item()

            bias_true.extend([int(i) for i in bias])
            bias_pred.extend([int(i) for i in torch.max(bias_output, 1)[1]])

            hyperp_true.extend([int(i) for i in hyperp.flatten()])
            hyperp_pred.extend([int(i) for i in hyperp_output.flatten().round()])

    bias_correct = int(np.equal(bias_true, bias_pred).sum())
    hyperp_correct = int(np.equal(hyperp_true, hyperp_pred).sum())
    n = len(bias_true)

    bias_labels = [data.fields['bias'].vocab.stoi[l]
                   for l in ['left', 'left-center', 'least', 'right-center', 'right']]

    print(confusion_matrix(bias_true, bias_pred, labels=bias_labels))
    print(confusion_matrix(hyperp_true, hyperp_pred))

    return total_loss / n, bias_correct / n, hyperp_correct / n


def get_iterator(dataset, batch_size, train=True):
    return Iterator(
        dataset, batch_size=batch_size, device=device,
        train=train, shuffle=train,
        sort=False
    )


class BiLSTM(nn.Module):
    def __init__(self, title_size, text_size, title_vectors, text_vectors):
        super(BiLSTM, self).__init__()

        self.dropout_in = nn.Dropout(0.5)

        title_embed_size = 300
        title_hidden_size = 50
        self.title_embeds = nn.Embedding(title_size, title_embed_size)
        self.title_embeds.weight.data.copy_(title_vectors)
        self.title_embeds.weight.requires_grad = False
        self.title_lstm = nn.LSTM(title_embed_size, title_hidden_size, dropout=0.4, num_layers=2, batch_first=True)  #

        text_embed_size = 300
        text_hidden_size = 50
        self.text_embeds = nn.Embedding(text_size, text_embed_size)
        self.text_embeds.weight.data.copy_(text_vectors)
        self.text_embeds.weight.requires_grad = False
        self.text_lstm = nn.LSTM(text_embed_size, text_hidden_size, dropout=0.4, num_layers=2, batch_first=True)  #

        hidden_size = 100
        self.linear1 = nn.Linear(title_hidden_size + text_hidden_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.bias_out = nn.Linear(hidden_size, 5)
        self.hyperp_out = nn.Linear(hidden_size, 1)

    def forward(self, X_titles, title_lengths, X_texts, text_lengths):
        # Titles
        X_titles = self.title_embeds(X_titles)
        X_titles = self.dropout_in(X_titles)
        X_titles = pack_padded_sequence(X_titles, title_lengths, batch_first=True, enforce_sorted=False)
        _, X_titles = self.title_lstm(X_titles)
        X_titles = F.relu(X_titles[0][0])

        # # Texts
        X_texts = self.text_embeds(X_texts)
        X_texts = self.dropout_in(X_texts)
        X_texts = pack_padded_sequence(X_texts, text_lengths, batch_first=True, enforce_sorted=False)
        _, X_texts = self.text_lstm(X_texts)
        X_texts = F.relu(X_texts[0][0])

        X = torch.cat((X_titles, X_texts), 1)
        # X = X_titles
        X = F.relu(self.linear1(X))
        # X = F.relu(self.linear2(X))

        X_bias = F.log_softmax(self.bias_out(X), dim=1)
        X_hyperp = torch.sigmoid(self.hyperp_out(X))
        return X_bias, X_hyperp


def train():
    model.train()
    total_loss, total, bias_correct, hyperp_correct = 0., 0, 0, 0
    start_time = time.time()

    train_iter = get_iterator(train_data, batch_size)

    for i, batch in enumerate(train_iter):
        titles, title_lengths = batch.title
        texts, text_lengths = batch.text
        # publishers = batch.publisher
        bias = batch.bias
        hyperp = batch.hyperp.reshape((-1, 1)).float()

        optimizer.zero_grad()
        bias_output, hyperp_output = model(titles, title_lengths, texts, text_lengths)
        loss = bias_criterion(bias_output, bias) + 0.5 * hyperp_criterion(hyperp_output, hyperp)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += len(bias)

        bias_correct += int(torch.max(bias_output, 1)[1].eq(bias).sum())
        hyperp_correct += int(hyperp_output.flatten().round().eq(hyperp.flatten()).sum())

        log_interval = 50
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            bias_acc = bias_correct / total
            hyperp_acc = hyperp_correct / total

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:} batches | '
                  'ms/batch {:5.2f} | '
                  'loss {:5.2f} | bias acc {:5.2f} | '
                  'hyperp acc {:5.2f}'.format(
                      epoch, i, len(train_iter),
                      elapsed * 1000 / log_interval,
                      cur_loss, bias_acc, hyperp_acc))

            total_loss, total, bias_correct, hyperp_correct = 0., 0, 0, 0
            start_time = time.time()


print(' > Loading data')
train_data, val_data = get_dataset('hyperp-training-grouped.csv.xz', full_training=False,
                                   random_valid=True, lower=False, vectors='glove.840B.300d')
print(' > Data loaded')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 64

bias_vocab = train_data.fields['bias'].vocab
title_vocab = train_data.fields['title'].vocab
text_vocab = train_data.fields['text'].vocab

model = BiLSTM(len(title_vocab), len(text_vocab), title_vocab.vectors, text_vocab.vectors).to(device)

bias_criterion = nn.NLLLoss()
hyperp_criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  #

best_val_loss = float("inf")
all_bias_acc, all_hyperp_acc = [], []
epochs = 30
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss, bias_acc, hyperp_acc = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | val bias acc {:5.2f} | val hyperp acc {:5.2f}'.format(
        epoch, (time.time() - epoch_start_time), val_loss, bias_acc, hyperp_acc))
    print('-' * 89)

    all_bias_acc.append(bias_acc)
    all_hyperp_acc.append(hyperp_acc)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
