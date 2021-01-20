import pandas as pd 
import os
import string
import re
import itertools
import random
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

def getData(
    file_path,
    sent_ind,
    tokens,
    tags
):
    data = pd.read_csv(
        file_path, error_bad_lines= False
    )
    tokens = data.groupby(sent_ind)[tokens].apply(list).values[:100]
    tags = data.groupby(sent_ind)[tags].apply(list).values[:100]

    return tokens, tags

def words2indices(origin, vocab):
    """ Transform a sentence or a list of sentences from str to int
    Args:
        origin: a sentence of type list[str], or a list of sentences of type list[list[str]]
        vocab: Vocab instance
    Returns:
        a sentence or a list of sentences represented with int
    """
    if isinstance(origin[0], list):
        result = [[vocab[w] for w in sent] for sent in origin]
    else:
        result = [vocab[w] for w in origin]
    return result

def generate_train_dev_dataset(
        filepath, sent_vocab, tag_vocab, train_proportion, 
        sent_ind,
        tokens,
        tags
    ):
    """ Read corpus from given file path and split it into train and dev parts
    Returns:
        train_data: data for training, list of tuples, each containing a sentence and corresponding tag.
        dev_data: data for development, list of tuples, each containing a sentence and corresponding tag.
    """
    sentences, tags = getData(
        filepath,
        sent_ind,
        tokens,
        tags
    )
    sentences = words2indices(sentences, sent_vocab)
    tags = words2indices(tags, tag_vocab)
    data = list(zip(sentences, tags))
    random.shuffle(data)
    n_train = int(len(data) * train_proportion)
    train_data, dev_data = data[: n_train], data[n_train:]
    return train_data, dev_data

def createDir(path_list):
    for _, path in enumerate(path_list):
        if not os.path.exists(path):
            os.mkdir(path)
    
def batch_iter(data, batch_size=32, shuffle=True):
    """ Yield batch of (sent, tag), by the reversed order of source length.
    Args:
        data: list of tuples, each tuple contains a sentence and corresponding tag.
        batch_size: batch size
        shuffle: bool value, whether to random shuffle the data
    """
    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        #order indices in order we want
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        #get values at those indices
        sentences = [x[0] for x in batch]
        tags = [x[1] for x in batch]
        yield sentences, tags

def pad(data, padded_token, device):
    """ pad data so that each sentence has the same length as the longest sentence
    Args:
        data: list of sentences, List[List[word]]
        padded_token: padded token
        device: device to store data
    Returns:
        padded_data: padded data, a tensor of shape (max_len, b)
        lengths: lengths of batches, a list of length b.
    """
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths

#TODO, find right tokenization
def tokenize(
    text
):
    separators = string.punctuation + string.whitespace
    separators_re = "|".join(re.escape(x) for x in separators)
    tokens = zip(re.split(separators_re, text), re.findall(separators_re, text))
    flattened = itertools.chain.from_iterable(tokens)
    cleaned = [x for x in flattened if x and not x.isspace()]
    return cleaned

def getConfidence(
    logits
):
    softmax = nn.Softmax(dim=1)
    return softmax(logits)