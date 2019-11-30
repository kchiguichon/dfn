import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from typing import List, Dict
from tensorflow.keras import models, optimizers

from models.neural_models import DAN, DFN, GRU
from util.model_util import save_model, load_model
from data import swap_key_values, load_eval_data, load_glove_embeddings, process_data, generate_batches

def eval(model: models.Model, eval_batches: List) -> tf.keras.Model:
    total_eval_loss = 0
    total_correct_predictions, total_predictions = 0, 0
    generator_tqdm = tqdm(eval_batches)
    for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
        logits = model(batch_inputs, training=False)
        loss_value = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(batch_labels, logits))
        total_eval_loss += loss_value
        batch_predictions = tf.math.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
        total_correct_predictions += tf.math.reduce_sum(tf.cast(batch_predictions == batch_labels, dtype=tf.int64))
        total_predictions += batch_labels.get_shape()[0]
        description = ("Average evaluation loss: %.2f Accuracy: %.2f "
                        % (total_eval_loss/(index+1), total_correct_predictions/total_predictions))
        generator_tqdm.set_description(description, refresh=False)
    average_loss = total_eval_loss / len(eval_batches)
    eval_accuracy = total_correct_predictions/total_predictions
    print('Final evaluation accuracy: %.4f loss: %.4f' % (eval_accuracy, average_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Script to train model on data.""")
    parser.add_argument('model', help='Path to pretrained model directory')
    parser.add_argument('--test', help='Path to evaluation data.', default=r'./data/test.csv')
    parser.add_argument('--labels', help='Path to label dictionary.', default=r'./data/answers.json')

    args = parser.parse_args()
    data, label_to_id = load_eval_data(args.test, args.labels)
    print('\nLoading test data...')
    model, model_config, vocab, reverse_vocab = load_model(args.model)
    test_X, test_Y, vocab, reverse_vocab = process_data(
        data, 
        label_to_id, 
        vocab=vocab, 
        vocab_size=model_config['vocab_size']
    )
    print('Test data loaded.')
    batches = generate_batches(test_X, test_Y, 32)
    print('Batches finished generating.')
    train_result = eval(model, batches)
