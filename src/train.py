import re
import os
import json
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, optimizers
from typing import List, Dict

from util.constants import PAD_TOKEN, UNK_TOKEN
from models.neural_models import DAN, DFN, GRU

LABEL_TO_ID = dict()
ID_TO_LABEL = dict()

def swap_key_values(dictionary):
    return dict([(value, key) for key, value in dictionary.items()])

def load_data(train_path, dev_path, labels_path):
    global LABEL_TO_ID, ID_TO_LABEL
    LABEL_TO_ID = json.load(open(labels_path, 'r', encoding='utf8'))
    ID_TO_LABEL = swap_key_values(LABEL_TO_ID)
    return {
        'train' : pd.read_csv(open(train_path, 'r'), quotechar='"'),
        'dev' : pd.read_csv(open(dev_path, 'r'), quotechar='"')
    }

def process_data(dataframe : pd.DataFrame, 
                 vocab=None,
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

    vocab : ``dict``
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
    global LABEL_TO_ID, ID_TO_LABEL
    if vocab is not None and (PAD_TOKEN not in vocab or UNK_TOKEN not in vocab):
        raise ValueError('Both {} token and {} token must be in vocabulary.'.format(PAD_TOKEN, UNK_TOKEN))
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
    labels = tf.keras.backend.map_fn(np.vectorize(lambda x: LABEL_TO_ID[x.decode('utf8')]), labels, dtype=tf.int64)
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

def train(model: models.Model,
          optimizer: optimizers.Optimizer,
          train_batches: List,
          validation_batches: List,
          num_epochs: int,
          serialization_dir: str = None) -> tf.keras.Model:
    global LABEL_TO_ID
    best_epoch_validation_accuracy = float("-inf")
    best_epoch_validation_loss = float("inf")
    regularization_lambda = 1e-5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")
        total_training_loss = 0
        total_correct_predictions, total_predictions = 0, 0
        generator_tqdm = tqdm(train_batches)
        for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
            with tf.GradientTape() as tape:
                logits = model(batch_inputs, training=True)
                loss_value = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(batch_labels, logits))
                regularization = regularization_lambda * tf.reduce_sum([tf.nn.l2_loss(x) for x in model.weights])
                grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_training_loss += loss_value
            batch_predictions = tf.math.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
            total_correct_predictions += tf.math.reduce_sum(tf.cast(batch_predictions == batch_labels, dtype=tf.int64))
            total_predictions += batch_labels.get_shape()[0]
            description = ("Average training loss: %.2f Accuracy: %.2f "
                           % (total_training_loss/(index+1), total_correct_predictions/total_predictions))
            generator_tqdm.set_description(description, refresh=False)
        average_training_loss = total_training_loss / len(train_batches)
        training_accuracy = total_correct_predictions/total_predictions

        total_validation_loss = 0
        total_correct_predictions, total_predictions = 0, 0
        generator_tqdm = tqdm(validation_batches)
        for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
            logits = model(batch_inputs, training=False)
            loss_value = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(batch_labels, logits))
            total_validation_loss += loss_value
            batch_predictions = tf.math.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
            total_correct_predictions += tf.math.reduce_sum(tf.cast(batch_predictions == batch_labels, dtype=tf.int64))
            total_predictions += batch_labels.get_shape()[0]
            description = ("Average validation loss: %.2f Accuracy: %.2f "
                           % (total_validation_loss/(index+1), total_correct_predictions/total_predictions))
            generator_tqdm.set_description(description, refresh=False)
        average_validation_loss = total_validation_loss / len(validation_batches)
        validation_accuracy = total_correct_predictions/total_predictions

        if validation_accuracy > best_epoch_validation_accuracy:
            if serialization_dir is not None:
                print("Model with best validation accuracy so far: %.2f. Saving the model."
                    % (validation_accuracy))
                model.save_weights(os.path.join(serialization_dir, f'model.ckpt'))
            best_epoch_validation_loss = average_validation_loss
            best_epoch_validation_accuracy = validation_accuracy

    metrics = {"training_loss": float(average_training_loss),
               "validation_loss": float(average_validation_loss),
               "training_accuracy": float(training_accuracy),
               "best_epoch_validation_accuracy": float(best_epoch_validation_accuracy),
               "best_epoch_validation_loss": float(best_epoch_validation_loss)}

    print("Best epoch validation accuracy: %.4f, validation loss: %.4f"
          %(best_epoch_validation_accuracy, best_epoch_validation_loss))

    return {"model": model, "metrics": metrics}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Script to train model on data.""")
    parser.add_argument('--train', help='Path to train data.', default='.\\data\\train.csv')
    parser.add_argument('--dev', help='Path to dev data.', default='.\\data\\dev.csv')
    parser.add_argument('--labels', help='Path to label dictionary.', default='.\\data\\answers.json')
    parser.add_argument('--embeddings', help='Path to embeddings', default='.\\data\\glove.6B\\glove.6B.50d.txt')
    parser.add_argument('--embed-dim', help='Size of embeddings', type=int, default=50)
    parser.add_argument('--batch-size', help='Size of training batches.', type=int, default=32)
    parser.add_argument('--vocab-size', help='Size of vocabulary to use.', type=int, default=15_000)
    parser.add_argument('--sequence-length', help='Maximum size of sequences to use.', type=int, default=200)
    parser.add_argument('--num-epochs', help='Number of epochs.', type=int, default=10)
    parser.add_argument('--num-layers', help='Number of layers.', type=int, default=4)
    parser.add_argument('--hidden-dim', help='Size of hidden representation vector.', type=int, default=150)
    args = parser.parse_args()
    data = load_data(args.train, args.dev, args.labels)
    train_data = data['train']
    validation_data = data['dev']
    print('\nLoading training data...')
    train_X, train_Y, vocab, reverse_vocab = process_data(train_data, vocab_size=args.vocab_size, max_tokens=args.sequence_length)
    print('Training data loaded.')
    print('\nLoading validation data...')
    validation_X, validation_Y, _, _ = process_data(validation_data, vocab=vocab, max_tokens=args.sequence_length)
    print('Validation data loaded.')
    print('\nGenerating batches...')
    train_batches = generate_batches(train_X, train_Y, args.batch_size)
    validation_batches = generate_batches(validation_X, validation_Y, args.batch_size)
    print('Batches finished generating.')

    optimizer = optimizers.Adam()

    model_config = {
        'vocab_size': args.vocab_size, 
        'embedding_dim': args.embed_dim, 
        'output_dim': len(LABEL_TO_ID), 
        'num_layers': args.num_layers, 
        'hidden_dim' : args.hidden_dim,
        'dropout': 0.2,
        'trainable_embeddings': True
    }

    model = DFN(**model_config)
    model.embeddings.assign(load_glove_embeddings(args.embeddings, args.embed_dim, reverse_vocab))

    train(model, optimizer, train_batches, validation_batches, args.num_epochs)
