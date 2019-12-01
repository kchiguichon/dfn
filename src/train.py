# Copyright 2019 Kenneth Chiguichon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from typing import List, Dict
from tensorflow.keras import models, optimizers

from models.neural_models import DAN, DFN, GRU
from util.model_util import save_model, load_model
from data import swap_key_values, load_data, load_glove_embeddings, process_data, generate_batches

def train(model: models.Model,
          optimizer: optimizers.Optimizer,
          train_batches: List,
          validation_batches: List,
          num_epochs: int,
          serialization_dir: str = None,
          config: Dict = None,
          vocab: Dict = None) -> tf.keras.Model:
    tensorboard_logs_path = os.path.join(serialization_dir, f'tensorboard_logs')
    tensorboard_writer = tf.summary.create_file_writer(tensorboard_logs_path)
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
                save_model(model, config, vocab, serialization_dir)
            best_epoch_validation_loss = average_validation_loss
            best_epoch_validation_accuracy = validation_accuracy

        with tensorboard_writer.as_default():
            tf.summary.scalar("loss/training", average_training_loss, step=epoch)
            tf.summary.scalar("loss/validation", average_validation_loss, step=epoch)
            tf.summary.scalar("accuracy/training", training_accuracy, step=epoch)
            tf.summary.scalar("accuracy/validation", validation_accuracy, step=epoch)
        tensorboard_writer.flush()

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
    parser.add_argument('--train', help='Path to train data.', default=r'./data/train.csv')
    parser.add_argument('--dev', help='Path to dev data.', default=r'./data/dev.csv')
    parser.add_argument('--labels', help='Path to label dictionary.', default=r'./data/answers.json')
    parser.add_argument('--pretrained-model', help='Path to pretrained model directory')
    parser.add_argument('--checkpoint-path', help='Path to save model checkpoints.', default=r'./serialization_dirs/default/')
    parser.add_argument('--model-type', help='Model to use for QA task.', choices=('DAN', 'DFN', 'GRU'), type=str.upper, default='DAN')
    parser.add_argument('--embeddings', help='Path to embeddings')
    parser.add_argument('--embed-dim', help='Size of embeddings', type=int, default=50)
    parser.add_argument('--batch-size', help='Size of training batches.', type=int, default=32)
    parser.add_argument('--vocab-size', help='Size of vocabulary to use.', type=int, default=15_000)
    parser.add_argument('--sequence-length', help='Maximum size of sequences to use.', type=int, default=200)
    parser.add_argument('--num-epochs', help='Number of epochs.', type=int, default=10)
    parser.add_argument('--num-layers', help='Number of layers.', type=int, default=4)
    parser.add_argument('--hidden-dim', help='Size of hidden representation vector.', type=int, default=-1)

    args = parser.parse_args()
    data, label_to_id = load_data(args.train, args.dev, args.labels)
    train_data = data['train']
    validation_data = data['dev']
    vocab = None
    model = None
    if args.pretrained_model is not None:
        model, model_config, vocab, reverse_vocab = load_model(args.pretrained_model)
    print('\nLoading training data...')
    train_X, train_Y, vocab, reverse_vocab = process_data(
        train_data, 
        label_to_id, 
        vocab=vocab, 
        vocab_size=args.vocab_size, 
        max_tokens=args.sequence_length
    )
    print('Training data loaded.')
    print('\nLoading validation data...')
    validation_X, validation_Y, _, _ = process_data(
        validation_data, 
        label_to_id, 
        vocab=vocab, 
        max_tokens=args.sequence_length
    )
    print('Validation data loaded.')
    print('\nGenerating batches...')
    train_batches = generate_batches(train_X, train_Y, args.batch_size)
    validation_batches = generate_batches(validation_X, validation_Y, args.batch_size)
    print('Batches finished generating.')

    optimizer = optimizers.Adam()
    
    if model is None:
        model_config = {
            'vocab_size': args.vocab_size, 
            'embedding_dim': args.embed_dim, 
            'output_dim': len(label_to_id), 
            'num_layers': args.num_layers, 
            'dropout': 0.2,
            'trainable_embeddings': True
        }

        if args.model_type == 'DAN':
            model_config['hidden_dim'] = args.embed_dim if args.hidden_dim == -1 else args.hidden_dim
            model = DAN(**model_config)
        elif args.model_type == 'DFN':
            model_config['hidden_dim'] = 150 if args.hidden_dim == -1 else args.hidden_dim
            model = DFN(**model_config)
        else:
            model_config['hidden_dim'] = args.sequence_length if args.hidden_dim == -1 else args.hidden_dim
            model = GRU(**model_config)

        if args.embeddings is not None:
            model.embeddings.assign(load_glove_embeddings(args.embeddings, args.embed_dim, reverse_vocab))
    
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    elif os.path.exists(os.path.join(args.checkpoint_path, f'tensorboard_logs')):
        shutil.rmtree(os.path.join(args.checkpoint_path, f'tensorboard_logs'))
    train_result = train(
        model, 
        optimizer, 
        train_batches, 
        validation_batches, 
        args.num_epochs, 
        args.checkpoint_path, 
        model_config, 
        vocab
    )
    model, metrics = train_result['model'], train_result['metrics']
    json.dump(metrics, open(os.path.join(args.checkpoint_path, f'metrics.json'), 'w', encoding='utf8'))
