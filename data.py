import os
import random
from collections import Counter

from sacremoses import MosesTokenizer, MosesPunctNormalizer
import pandas as pd
import numpy as np
from torchtext.data import TabularDataset, Field
from tqdm import tqdm


def clean_text(txt, flat=True, line_blacklist={''}, token_blacklist=set()):
    txt = ' '.join([l.strip() for l in txt.split('\n') if l.strip() not in line_blacklist])

    if len(txt) == 0:
        return '.'

    if '?' in txt and txt.index('?') != len(txt) - 1:
        first_idx = txt.index('?')
        if txt[first_idx + 1] == ' ':
            txt = txt.replace('?', '\'')

    for t in token_blacklist:
        txt = txt.replace(t, '')

    if len(txt) == 0:
        return '.'

    return txt


def get_line_blacklist(df, n=200):
    lines = [(p, l.strip()) for p, t in df[['publisher', 'text']].itertuples(index=False) for l in t.split('\n')]
    line_counts = Counter(lines)
    return {l[1] for l, _ in line_counts.most_common(n)}


def get_token_blacklist(df, n=100, min_pubf=50):
    print(' :: Counting tokens')
    token_counts = {(i, p, l) for i, p, t in df[['id', 'publisher', 'text']
                                                ].itertuples(index=False) for l in t.split()}
    token_tfs = Counter([(p, l) for _, p, l in token_counts])

    print(' :: Counting publishers')
    pub_dfs = {t: 0 for (_, t), _ in set(token_tfs.items())}
    for (pub, t), _ in token_tfs.items():
        pub_dfs[t] += 1

    pubf = Counter(df.publisher)

    print(' :: Calculating scores')
    token_tfidfs = {}
    for (pub, tok), tf in token_tfs.items():
        if pubf[pub] > min_pubf:
            token_tfidfs[(pub, tok)] = tf / pubf[pub] / pub_dfs[tok]

    sorted_tfidfs = sorted(token_tfidfs, key=token_tfidfs.get, reverse=True)[:n]
    return {t for _, t in sorted_tfidfs}


def split_df(df):
    np.random.seed(623)

    n_valid_publishers = {
        'left': 15,
        'left-center': 6,
        'least': 2,
        'right-center': 1,
        'right': 16
    }

    largest_publishers = set(df.publisher.value_counts()[:5].index)

    valid_publishers = []
    for bias in df.bias.unique():
        pubs = list(set(df.publisher[df.bias == bias].drop_duplicates()) - largest_publishers)
        np.random.shuffle(pubs)
        valid_publishers.extend(pubs[:n_valid_publishers[bias]])

    train_df = df.loc[~df.publisher.isin(valid_publishers)]
    valid_df = df.loc[df.publisher.isin(valid_publishers)]
    return train_df, valid_df


def prepare_csv(path, cleanup=False):
    if path.endswith('.csv'):
        return path

    df = pd.read_csv(path, compression='xz', sep='\t', encoding='utf-8',
                     usecols=['id', 'hyperp', 'bias', 'publisher', 'title', 'text']).dropna()

    print(' > Cleaning up data')
    df.bias = df.bias.astype('category')
    df.publisher = df.publisher

    print(' > Initializing Moses')
    mt = MosesTokenizer()
    mn = MosesPunctNormalizer()

    line_blacklist, token_blacklist = None, None
    if cleanup:
        print(' > Generating line blacklist')
        line_blacklist = get_line_blacklist(df)
        print(' > Generating token blacklist')
        token_blacklist = get_token_blacklist(df)

    tqdm.pandas()
    print(' > Processing titles')
    df.title = df.title.apply(clean_text).apply(mn.normalize).apply_progress(mt.tokenize, return_str=True)

    print(' > Processing texts')
    df.text = df.text.apply(clean_text, line_blacklist=line_blacklist, token_blacklist=token_blacklist).apply_progress(
        mn.normalize).apply_progress(mt.tokenize, return_str=True)

    new_path = path.replace('.xz', '')
    df.to_csv(new_path, index=False)

    return new_path


def split_csv(full_path):
    train_path = full_path.replace('.csv', '.train.csv')
    val_path = full_path.replace('.csv', '.val.csv')

    if os.path.isfile(train_path) and os.path.isfile(val_path):
        return train_path, val_path

    df = pd.read_csv(full_path)

    train_df, valid_df = split_df(df)
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(val_path, index=False)

    return train_path, val_path


def get_dataset(path, val_path=None, full_training=False, random_valid=False, max_length=256, lower=True, vectors=None):
    full_path = prepare_csv(path, cleanup=True)
    if val_path is not None:
        val_path = prepare_csv(val_path)

    if full_training:
        train_path = full_path
    else:
        train_path, val_path = split_csv(full_path)

    hyperp_field = Field(sequential=False, use_vocab=False, preprocessing=lambda x: int(x == 'True'))
    bias_field = Field(use_vocab=True, sequential=False, is_target=True, unk_token=None)
    publisher_field = Field(use_vocab=True, sequential=False)
    title_field = Field(lower=lower, fix_length=max_length, include_lengths=True,
                        use_vocab=True, sequential=True, batch_first=True)
    text_field = Field(lower=lower, fix_length=max_length, include_lengths=True,
                       use_vocab=True, sequential=True, batch_first=True)

    data_kwargs = {
        'format': 'csv',
        'skip_header': True,
        'fields': [
            ('id', None),
            ('hyperp', hyperp_field),
            ('bias', bias_field),
            ('publisher', publisher_field),
            ('title', title_field),
            ('text', text_field),
        ]
    }

    if random_valid:
        data = TabularDataset(path=full_path, **data_kwargs)
        random.seed(42)
        train, val = data.split(0.8, stratified=True, strata_field='bias', random_state=random.getstate())
    else:
        train, val = TabularDataset.splits(path='.', train=train_path, validation=val_path, **data_kwargs)

    bias_field.build_vocab(train)
    publisher_field.build_vocab(train)
    title_field.build_vocab(train, vectors=vectors)
    text_field.build_vocab(train, vectors=vectors)
    return train, val
