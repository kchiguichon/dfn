import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy import fftpack, ndimage
import matplotlib.pyplot as plt
import random

class DAN(models.Model):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 output_dim: int, 
                 num_layers: int,
                 hidden_dim: int, 
                 dropout: float = 0.2,
                 trainable_embeddings: bool = True):
        super(DAN, self).__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.embeddings = tf.Variable(tf.random.normal([vocab_size, embedding_dim]), trainable=trainable_embeddings)
        for i in range(self.num_layers):
            name = 'dense' + str(i+1)
            setattr(self, name, layers.Dense(hidden_dim, activation='tanh', name=name))
        self.classifier = layers.Dense(output_dim)

    def call(self, batch_data: tf.Tensor, training=False) -> tf.Tensor:
        sequence_mask = tf.cast(batch_data != 0, dtype=tf.float32)
        logits = tf.nn.embedding_lookup(self.embeddings, batch_data)
        if training:
            # Word dropout
            dropout_mask = tf.cast(tf.random.uniform(batch_data.get_shape()) >= self.dropout_prob, dtype=tf.float32) * sequence_mask
            divisor = tf.expand_dims(tf.reduce_sum(dropout_mask, 1, True), [-1])
            logits = tf.squeeze(tf.divide(
                tf.reduce_sum(logits * tf.expand_dims(dropout_mask, [-1]), 1, True),
                tf.where(divisor == 0, tf.ones_like(divisor), divisor)
            ))
        else:
            if sequence_mask is not None:
                divisor = tf.expand_dims(tf.reduce_sum(sequence_mask, 1, True), [-1])
                sequence_mask = tf.cast(tf.expand_dims(sequence_mask, [-1]), dtype=tf.float32)
                inputs = logits * sequence_mask
            else:
                divisor = tf.expand_dims(tf.reduce_sum(tf.ones_like(logits), 1, True), [-1])
                inputs = logits
            logits = tf.squeeze(tf.divide(tf.reduce_sum(inputs, 1, True), divisor))
        
        for i in range(self.num_layers):
            logits = getattr(self, 'dense' + str(i+1))(logits)
        logits = self.classifier(logits)
        return logits

class DFN(models.Model):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 output_dim: int, 
                 num_layers: int,
                 hidden_dim: int, 
                 dropout: float = 0.2,
                 trainable_embeddings: bool = True):
        super(DFN, self).__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout

        def swish(inputs):
            return inputs * tf.math.sigmoid(0.7 * inputs)
        tf.keras.utils.get_custom_objects().update({'swish' : layers.Activation(swish)})

        self.embeddings = tf.Variable(tf.random.normal([vocab_size, embedding_dim]), trainable=trainable_embeddings)
        for i in range(self.num_layers):
            dense_name, batch_norm_name = 'dense' + str(i+1), 'batch_norm' + str(i+1)
            setattr(self, dense_name, layers.Dense(hidden_dim, activation='swish', name=dense_name))
            setattr(self, batch_norm_name, layers.BatchNormalization(name=batch_norm_name))
        self.classifier = layers.Dense(output_dim)

    def call(self, batch_data: tf.Tensor, training=False) -> tf.Tensor:
        sequence_mask = tf.cast(tf.expand_dims(batch_data != 0, [-1]), dtype=tf.float32)
        logits = tf.nn.embedding_lookup(self.embeddings, batch_data) * sequence_mask

        if training:
            dropout_mask = tf.cast(tf.random.uniform(batch_data.get_shape()) >= self.dropout_prob, dtype=tf.float32)
            logits *= tf.expand_dims(dropout_mask, -1)

        x = tf.signal.rfft(logits)
        y = tf.transpose(x, [0, 2, 1])

        x = tf.reduce_sum(x, 1)
        y = tf.reduce_sum(y, 1)

        x = tf.signal.irfft(x)
        y = tf.signal.irfft(y)

        logits = tf.concat([x, y], -1)
        for i in range(self.num_layers):
            logits = getattr(self, 'dense' + str(i+1))(logits)
            logits = getattr(self, 'batch_norm' + str(i+1))(logits, training)
        logits = self.classifier(logits)
        return logits

class GRU(models.Model):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 output_dim: int, 
                 num_layers: int,
                 hidden_dim:int, 
                 dropout: float = 0.2,
                 trainable_embeddings: bool = True):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.embeddings = tf.Variable(tf.random.normal([vocab_size, embedding_dim]), trainable=trainable_embeddings)
        for i in range(self.num_layers):
            name = 'gru' + str(i+1)
            if i < num_layers -1:
                setattr(self, name, layers.GRU(hidden_dim, activation='tanh', return_sequences=True, name=name))
            else:
                setattr(self, name, layers.GRU(hidden_dim, activation='tanh', name=name))
        self.classifier = layers.Dense(output_dim)

    def call(self, batch_data: tf.Tensor, training=False) -> tf.Tensor:
        sequence_mask = batch_data != 0
        logits = tf.nn.embedding_lookup(self.embeddings, batch_data)
        for i in range(self.num_layers):
            logits = getattr(self, 'gru' + str(i+1))(logits, mask=sequence_mask)
        logits = self.classifier(logits)
        return logits

