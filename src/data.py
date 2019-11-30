import re
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from typing import List, Dict
from util.constants import PAD_TOKEN, UNK_TOKEN

def swap_key_values(dictionary):
    return dict([(value, key) for key, value in dictionary.items()])

def load_data(train_path, dev_path, labels_path):
    return ({
        'train' : pd.read_csv(open(train_path, 'r'), quotechar='"'),
        'dev' : pd.read_csv(open(dev_path, 'r'), quotechar='"')
    }, json.load(open(labels_path, 'r', encoding='utf8')))
    
def load_eval_data(eval_path, labels_path):
    return pd.read_csv(open(eval_path, 'r'), quotechar='"'), json.load(open(labels_path, 'r', encoding='utf8'))

def process_data(dataframe : pd.DataFrame,
                 label_to_id: Dict,
                 vocab : Dict = None,
                 vocab_size : int = 10_000, 
                 max_tokens : int = 200, 
                 max_token_size : int = 40) -> (tf.Tensor, tf.Tensor, dict, dict):
    """
    Will take a dataframe read from ``load_data`` and return indexed data, labels, and vocabulary tables
    for that dataset.

    Parameters
    ---------- 
    dataframe : ``pd.DataFrame``
        A pandas dataframe containing data to be processed, and from which to build vocabulary.

    label_to_id : ``Dict``
        A dictionary mapping labels to corresponding ids.

    vocab : ``Dict``
        (Optional) Dictionary mapping tokens to indices. 

    vocab_size : ``int``
        (Optional) If vocab is ``None`` then this denotes the maximum size of the vocabulary 
        including padding and unknown tokens.

    max_tokens : ``int``
        (Optional) Maximum number of tokens (aka. words) per sequence. Sequences will be padded to max_tokens if
        their length is less than ``max_tokens``.

    max_token_size : ``int``
        (Optional) Maximum size of an individual token (i.e. how many characters in a token/word).

    Returns:
    --------
    data : ``tf.Tensor``
        Tensor containing indexed sequences of tokens.

    labels : ``tf.Tensor``
        Tensor containing label for each sequence in ``data``.

    vocab : ``dict``
        Dictionary mapping tokens to indices.

    reverse_vocab : ``dict``
        Dictionary mapping indices to tokens.
    """
    if vocab is not None:
        if PAD_TOKEN not in vocab or UNK_TOKEN not in vocab:
            raise ValueError('Both {} token and {} token must be in vocabulary.'.format(PAD_TOKEN, UNK_TOKEN))
        else:
            vocab_size = len(vocab)
    def _process_data_helper(text):
        # Tokenize text data
        tokens = re.findall(r'\w+|[^\w\s]', re.sub(r'[|]{3}', '', text.strip().lower()))[:max_tokens]
        # Padding
        tokens += [PAD_TOKEN] * (max_tokens - len(tokens))
        return np.asarray(tokens).astype('<U{}'.format(max_token_size))
    # Tokenize data and labels
    data = tf.convert_to_tensor(dataframe['Text'].apply(_process_data_helper))
    labels = tf.convert_to_tensor(dataframe['Answer'].apply(lambda x: np.array(re.sub(r'\s+', '_', x))))
    if vocab is None:
        # Build vocab
        counts = np.unique(data, return_counts=True)
        counts = [x[counts[0] != PAD_TOKEN.encode('utf8')] for x in counts]
        top_words = counts[0][np.argsort(counts[1])[:vocab_size-2:-1]]
        top_words = [byte_string.decode('utf8') for byte_string in top_words]
        vocab = dict(zip([PAD_TOKEN, UNK_TOKEN]+top_words, range(min(vocab_size, len(top_words)+2))))
        reverse_vocab = dict(zip(range(min(vocab_size, len(top_words)+2)), [PAD_TOKEN, UNK_TOKEN]+top_words))
    else:
        reverse_vocab = swap_key_values(vocab)
    # Map tokens to indices
    def index_lookup(token):
        token = token.decode('utf8')
        return vocab[token] if token in vocab else vocab[UNK_TOKEN]
    data = tf.keras.backend.map_fn(np.vectorize(index_lookup), data, dtype=tf.int32)
    labels = tf.keras.backend.map_fn(np.vectorize(lambda x: label_to_id[x.decode('utf8')]), labels, dtype=tf.int64)
    return data, labels, vocab, reverse_vocab

def generate_batches(X, Y, batch_size):
    return list(zip(*[[data[i:i+batch_size] for i in range(0, len(data), batch_size)] for data in [X,Y]]))

def load_glove_embeddings(embeddings_txt_file: str,
                          embedding_dim: int,
                          vocab_id_to_token: Dict[int, str]) -> np.ndarray:
    """
    Given a vocabulary (mapping from index to token), this function builds
    an embedding matrix of vocabulary size in which ith row vector is an
    entry from pretrained embeddings (loaded from embeddings_txt_file).
    """
    tokens_to_keep = set(vocab_id_to_token.values())
    vocab_size = len(vocab_id_to_token)

    embeddings = {}
    print("\nReading pretrained embedding file.")
    with open(embeddings_txt_file, encoding='utf8') as file:
        for line in tqdm(file):
            line = str(line).strip()
            token = line.split(' ', 1)[0]
            if not token in tokens_to_keep:
                continue
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                raise Exception(f"Pretrained embedding vector and expected "
                                f"embedding_dim do not match for {token}.")
                continue
            vector = np.asarray(fields[1:], dtype='float32')
            embeddings[token] = vector

    # Estimate mean and std variation in embeddings and initialize it random normally with it
    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    embedding_matrix = np.random.normal(embeddings_mean, embeddings_std,
                                        (vocab_size, embedding_dim))
    embedding_matrix = np.asarray(embedding_matrix, dtype='float32')

    for idx, token in vocab_id_to_token.items():
        if token in embeddings:
            embedding_matrix[idx] = embeddings[token]

    return embedding_matrix
