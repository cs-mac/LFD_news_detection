import time
import sys

from torchtext.data import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from sklearn.metrics import classification_report, confusion_matrix

import numpy as np

from data import get_dataset

BATCH_SIZE = 64

bias_criterion = nn.NLLLoss()
hyperp_criterion = nn.BCELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, data):
    model.eval()

    total_loss = 0.
    bias_true, bias_pred = [], []
    hyperp_true, hyperp_pred = [], []

    with torch.no_grad():
        test_iter = get_iterator(data, BATCH_SIZE)
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

    # print(confusion_matrix(bias_true, bias_pred, labels=bias_labels))
    # print(confusion_matrix(hyperp_true, hyperp_pred))

    bias_names = data.fields['bias'].vocab.itos

    hyperp_true = ['true' if bias_names[i] in {'left', 'right'} else 'false' for i in bias_true]
    hyperp_pred = ['true' if bias_names[i] in {'left', 'right'} else 'false' for i in bias_pred]

    print('\n > Hyperp results:')
    print(classification_report(hyperp_true, hyperp_pred))

    print('\n > Bias results:')
    print(classification_report(bias_true, bias_pred, target_names=bias_names))

    print('\n > Joint results:')
    joint_true = ['{} {}'.format(h, bias_names[i]) for i, h in zip(bias_true, hyperp_true)]
    joint_pred = ['{} {}'.format(h, bias_names[i]) for i, h in zip(bias_pred, hyperp_pred)]
    print(classification_report(joint_true, joint_pred))

    print('', confusion_matrix(bias_true, bias_pred))

    # print(classification_report(bias_true, bias_pred, target_names=bias_names, labels=bias_labels))

    return total_loss / n, bias_correct / n, hyperp_correct / n


def save_predictions(model, data, filename):
    all_ids = []
    all_preds = []
    bias_itos = data.fields['bias'].vocab.itos

    with torch.no_grad():
        test_iter = get_iterator(data, BATCH_SIZE)
        for batch in test_iter:
            ids = batch.id
            titles, title_lengths = batch.title
            texts, text_lengths = batch.text
            texts, text_lengths = batch.text

            bias_output, _ = model(titles, title_lengths, texts, text_lengths)
            all_preds.extend([bias_itos[int(i)] for i in torch.max(bias_output, 1)[1]])
            all_ids.extend([int(i) for i in ids])

    lines = []
    for i, pred in zip(all_ids, all_preds):
        hyperp = 'true' if pred in {'left', 'right'} else 'false'
        lines.append('{} {} {}\n'.format(i, hyperp, pred))

    with open(filename, 'w') as f:
        f.writelines(lines)


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


def train(model, train_data, optimizer, epoch):
    model.train()
    total_loss, total, bias_correct, hyperp_correct = 0., 0, 0, 0
    start_time = time.time()

    train_iter = get_iterator(train_data, BATCH_SIZE)

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


def run_training(model, train_data, test_data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  #

    best_val_loss = float("inf")
    all_bias_acc, all_hyperp_acc = [], []
    epochs = 30
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, train_data, optimizer, epoch)
        val_loss, bias_acc, hyperp_acc = evaluate(model, test_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | val bias acc {:5.2f} | val hyperp acc {:5.2f}'.format(
            epoch, (time.time() - epoch_start_time), val_loss, bias_acc, hyperp_acc))
        print('-' * 89)

        all_bias_acc.append(bias_acc)
        all_hyperp_acc.append(hyperp_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    print(' > Finished training, saving best model to rnn-model.pt')
    torch.save(best_model, 'rnn-model.pt')


def main():
    if len(sys.argv) < 3:
        print('Usage: python3 rnn.py <trainset> <testset> <model>')
        exit()

    train_path, test_path = sys.argv[1], sys.argv[2]

    test_path = None
    if len(sys.argv) >= 2:
        test_path = sys.argv[2]

    model_path = None
    if len(sys.argv) > 3:
        model_path = sys.argv[3]

    print(' > Loading data')
    train_data, test_data = get_dataset(train_path, test_path, full_training=True,
                                        random_valid=False, lower=False, vectors='glove.840B.300d')
    print(' > Data loaded')

    title_vocab = train_data.fields['title'].vocab
    text_vocab = train_data.fields['text'].vocab

    if model_path is not None:
        print(' > Loading pretrained model')
        model = torch.load(model_path, map_location=device)
    else:
        print(' > Training model')
        model = BiLSTM(len(title_vocab), len(text_vocab), title_vocab.vectors, text_vocab.vectors).to(device)
        run_training(model, train_data, test_data)

    print(' > Evaluating')
    evaluate(model, test_data)

    print(' > Exporting predictions')
    save_predictions(model, test_data, 'test-predictions.csv')


if __name__ == '__main__':
    main()
