import copy

import numpy as np
import tensorflow as tf


class RNN(tf.keras.layers.Layer):

    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size, window_size, **kwargs):

        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.encoder =tf.keras.layers.GRU(
            units=self.hidden_size,
            return_sequences=True,
            return_state=True
        )

        self.decoder = tf.keras.layers.GRU(
            units=self.hidden_size,
            return_sequences=True,
            return_state=True
        )

        self.src_embedding = tf.keras.layers.Embedding(
            input_dim=self.src_vocab_size,
            output_dim=self.hidden_size
        )

        self.tgt_embedding = tf.keras.layers.Embedding(
            input_dim=self.tgt_vocab_size,
            output_dim=self.hidden_size
        )

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(units=int(2 * self.tgt_vocab_size), activation='relu'),
            tf.keras.layers.Dense(units=self.tgt_vocab_size)
        ])

    def encode(self, src_inputs):

        src_embeddings = self.src_embedding(src_inputs)
        encoder_output, encoder_state = self.encoder(src_embeddings)
        return encoder_output, encoder_state

    def decode(self, tgt_inputs, encoder_states):
        tgt_embedings = self.tgt_embedding(tgt_inputs)
        decoder_output, decoder_state = self.decoder(tgt_embedings, initial_state=encoder_states)
        logits = self.classifier(decoder_output)
        return logits

    def get_embedding(self, words):
        return self.tgt_embedding(words)

    def call(self, src_inputs, tgt_inputs, src_padding_mask=None, tgt_padding_mask=None):

        src_embedings = self.src_embedding(src_inputs)

        encoder_output, encoder_state = self.encoder(src_embedings)

        tgt_embedings = self.tgt_embedding(tgt_inputs)

        decoder_output, decoder_state = self.decoder(tgt_embedings, initial_state=encoder_state)

        logits = self.classifier(decoder_output)

        return logits


if __name__ == "__main__":
    pass