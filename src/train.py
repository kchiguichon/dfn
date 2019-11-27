import re
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from util.constants import PAD_TOKEN, UNK_TOKEN

def load_data(train_path, dev_path):
    return {
        'train' : pd.read_csv(open(train_path, 'r'), quotechar='"'),
        'dev' : pd.read_csv(open(dev_path, 'r'), quotechar='"')
    }

def process_data(df, vocab_size=10000, max_tokens=200, max_token_size=40):
    def _process_data_helper(text):
        # Tokenize text data
        tokens = re.findall(r'\w+|[^\w\s]', re.sub(r'[|]{3}', '', text.strip().lower()))[:max_tokens]
        # Padding
        tokens += [PAD_TOKEN] * (max_tokens - len(tokens))
        return np.asarray(tokens).astype('<U{}'.format(max_token_size))
    # Tokenize data and labels
    data = tf.convert_to_tensor(df['Text'].apply(_process_data_helper))
    labels = tf.convert_to_tensor(df['Answer'].apply(lambda x: np.array(re.sub(r'\s+', '_', x))))
    # Build vocab
    counts = np.unique(data, return_counts=True)
    counts = [x[counts[0] != PAD_TOKEN.encode('utf8')] for x in counts]
    top_words = counts[0][np.argsort(counts[1])[:vocab_size-2:-1]]
    top_words = [byte_string.decode('utf8') for byte_string in top_words]
    vocab = dict(zip([PAD_TOKEN, UNK_TOKEN]+top_words, range(min(vocab_size, len(top_words)+2))))
    reverse_vocab = dict(zip(range(min(vocab_size, len(top_words)+2)), [PAD_TOKEN, UNK_TOKEN]+top_words))
    # Map tokens to indeces
    def indexer(token):
        token = token.decode('utf8')
        return vocab[token] if token in vocab else vocab[UNK_TOKEN]
    data = tf.keras.backend.map_fn(np.vectorize(indexer), data, dtype=tf.int32)
    return data, labels, vocab, reverse_vocab

def generate_train_batches(train_data, batch_size):
    pass

def generate_validation_instances(validation_data):
    pass

def train(*args, **kwargs):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Script to train model on data.
    """)

    parser.add_argument('--train', help='Path to train data.', default='.\\data\\train.csv')
    parser.add_argument('--dev', help='Path to dev data.', default='.\\data\\dev.csv')
    parser.add_argument('--batch-size', help='Size of training batches',type=int, default=32)
    args = parser.parse_args()
    data = load_data(args.train, args.dev)
    train_data = generate_train_batches(data['train'], args.batch_size)
    validation_data = generate_validation_instances(data['dev'])

    train(train_data, validation_data)
