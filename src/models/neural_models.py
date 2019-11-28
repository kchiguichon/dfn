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
                 dropout: float = 0.2,
                 trainable_embeddings: bool = True):
        super(DAN, self).__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.embeddings = tf.Variable(tf.random.normal([vocab_size, embedding_dim]), trainable=trainable_embeddings)
        for i in range(self.num_layers):
            name = 'dense' + str(i+1)
            setattr(self, name, layers.Dense(embedding_dim, activation='relu', name=name))
        self.classifier = layers.Dense(output_dim)

    def call(self, batch_data: tf.Tensor, training=False) -> tf.Tensor:
        sequence_mask = tf.cast(batch_data != 0, dtype=tf.float32)
        vector_sequence = tf.nn.embedding_lookup(self.embeddings, batch_data)
        if training:
            # Word dropout
            dropout_mask = tf.cast(tf.random.uniform(batch_data.get_shape()) >= self.dropout_prob, dtype=tf.float32) * sequence_mask
            divisor = tf.expand_dims(tf.reduce_sum(dropout_mask, 1, True), [-1])
            logits = tf.squeeze(tf.divide(
                tf.reduce_sum(vector_sequence * tf.expand_dims(dropout_mask, [-1]), 1, True),
                tf.where(divisor == 0, tf.ones_like(divisor), divisor)
            ))
        else:
            if sequence_mask is not None:
                divisor = tf.expand_dims(tf.reduce_sum(sequence_mask, 1, True), [-1])
                sequence_mask = tf.cast(tf.expand_dims(sequence_mask, [-1]), dtype=tf.float32)
                inputs = vector_sequence * sequence_mask
            else:
                divisor = tf.expand_dims(tf.reduce_sum(tf.ones_like(vector_sequence), 1, True), [-1])
                inputs = vector_sequence
            logits = tf.squeeze(tf.divide(tf.reduce_sum(inputs, 1, True), divisor))
        
        for i in range(self.num_layers):
            logits = getattr(self, 'dense' + str(i+1))(logits)
        logits = self.classifier(logits)
        return logits
