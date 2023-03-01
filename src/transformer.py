import tensorflow as tf
import numpy as np


class Transformer(tf.keras.Model):

    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size, window_size, nhead, **kwargs):
        super().__init__(**kwargs)
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO: Define image and positional encoding, transformer decoder, and classification layers

        self.encoder = TransformerBlock(
            self.hidden_size,
            nhead=nhead,
            type='encoder'
        )
        self.decoder = TransformerBlock(
            self.hidden_size,
            nhead=nhead,
            type='decoder'
        )

        self.src_encoding = PositionalEncoding(
            self.src_vocab_size,
            self.hidden_size,
            self.window_size
        )

        # Define positional encoding to embed and offset layer for language:
        self.tgt_encoding = PositionalEncoding(
            self.tgt_vocab_size,
            self.hidden_size,
            self.window_size
        )

        # Define classification layer (logits)
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(units=int(2 * self.tgt_vocab_size), activation='relu'),
            tf.keras.layers.Dense(units=self.tgt_vocab_size)
        ])

    def call(self, src_inputs, tgt_inputs, src_padding_mask=None, tgt_padding_mask=None):
        # TODO:
        # 1) Embed the src_inputs into a vector (HINT IN NOTEBOOK)
        # 2) Pass the tgt_inputs through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits
        probs = None

        src_embeddings = self.src_encoding(src_inputs)

        tgt_embeddings = self.tgt_encoding(tgt_inputs)

        encoder_output = self.encoder(src_embeddings, src_padding_mask)

        decoder_output = self.decoder(tgt_embeddings, encoder_output)

        probs = self.classifier(decoder_output)

        return probs

    def encode(self, src_inputs):
        src_embedding = self.src_encoding(src_inputs)
        return self.encoder.encode(src_embedding)

    def decode(self, tgt_inputs, encoder_output):
        tgt_embedding = self.tgt_encoding(tgt_inputs)
        output = self.decoder.decode(tgt_embedding, encoder_output)
        return self.classifier(output)

    def get_embedding(self, words):
        return self.tgt_encoding.embedding(words)

class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        """
        Computes attention given key and query matrices.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """

        #         print(inputs)

        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys = K.get_shape()[1]  # window size of keys

        #         print(window_size_queries)

        ## Fill triangle below diagonal of matrix with negative infinity and top part with 0.
        ## This helps to avoid over-contribution, since adjacency matrix is symmetric across diagonal.
        ## Tile this upward to be compatible with addition against computed attention scores.
        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]),
                             [tf.shape(input=K)[0], 1, 1])


        attention_logits = tf.matmul(Q, K, transpose_b=True)
        #         score = tf.tensordot(Q, tf.transpose(K), axes=[1,2])
        #         score = score / tf.sqrt(3.0)

        #         dk = tf.cast(tf.shape(K)[-1], tf.float32)
        dk = tf.cast(tf.shape(K)[1], tf.float32)
        #         https://edstem.org/us/courses/27644/discussion/2171270?answer=5009608

        attention_logits = attention_logits / tf.math.sqrt(dk)

        #         print("score:",score)
        #         print(atten_mask)
        if self.use_mask == True:
            #             score = tf.add(score, atten_mask)
            attention_logits = tf.add(attention_logits, atten_mask)
        #             attention_logits = attention_logits + atten_mask

        attention_weights = tf.nn.softmax(attention_logits, axis=-1)

        return attention_weights


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        # TODO:
        # Initialize the weight matrices for K, V, and Q.
        # They should be able to multiply an input_size vector to produce an output_size vector

        self.K = tf.Variable(tf.random.normal(shape=[input_size, output_size], stddev=0.01, dtype=tf.float32),
                             trainable=True)
        self.V = tf.Variable(tf.random.normal(shape=[input_size, output_size], stddev=0.01, dtype=tf.float32),
                             trainable=True)
        self.Q = tf.Variable(tf.random.normal(shape=[input_size, output_size], stddev=0.01, dtype=tf.float32),
                             trainable=True)

        self.attention_matrix = AttentionMatrix(use_mask=self.use_mask)

    #         self.K = self.add_weight()

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        # TODO:
        # - You will need to use tf.tensordot for this.
        # - Call your AttentionMatrix layer with the keys and queries.
        # - Apply the attention matrix to the values.
        # print("inputs_for_keys:", inputs_for_keys)
        # print('self K', self.K)
        K = tf.tensordot(inputs_for_keys, self.K, axes=1)
        V = tf.tensordot(inputs_for_values, self.V, axes=1)
        Q = tf.tensordot(inputs_for_queries, self.Q, axes=1)

        attention_matrix = self.attention_matrix([K, Q])
        attention = tf.matmul(attention_matrix, V)

        return attention


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, nhead, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)

        self.heads = [AttentionHead(input_size=emb_sz, output_size=emb_sz // nhead, is_self_attention=use_mask)
                      for _ in range(nhead)
                      ]

        self.emb_sz = emb_sz

        self.dense = tf.keras.layers.Dense(units=emb_sz)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        This functions runs a multiheaded attention layer.

        Requirements:
            - Splits data for 3 different heads of size embed_sz/3
            - Create three different attention heads
            - Concatenate the outputs of these heads together
            - Apply a linear layer

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        attns = [head(inputs_for_keys, inputs_for_values, inputs_for_queries) for head in self.heads]

        att = tf.concat(tuple(attns), axis=2)

        att = self.dense(att)

        return att


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, nhead=3, type='decoder', **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        # TODO:
        # 1) Define the Feed Forward, self-attention, encoder-decoder-attention, and layer normalization layers
        # 2) use multiheaded attention

        self.type = type
        self.ff_layer = tf.keras.layers.Dense(units=emb_sz)

        self.self_atten         = MultiHeadedAttention(emb_sz, True, nhead=nhead)

        if self.type == 'decoder':
            self.cross_atten        = MultiHeadedAttention(emb_sz, False, nhead=nhead)

        self.layer_norm = tf.keras.layers.LayerNormalization()


    @tf.function
    def call(self, inputs, context_sequence):
        if self.type == 'decoder':
            return self.decode(inputs, context_sequence)
        elif self.type == 'encoder':
            return self.encode(inputs)
        else:
            raise Exception("Invalid transformer block type")

    def encode(self, inputs):
        attn = self.self_atten(inputs, inputs, inputs)
        attn = self.layer_norm(inputs + attn)
        ff = self.ff_layer(attn)
        ff = self.layer_norm(attn + ff)
        return ff

    def decode(self, inputs, context_sequence):
        """
        This functions calls a transformer block.

        TODO:
        1) compute MASKED attention on the inputs
        2) residual connection and layer normalization
        3) computed UNMASKED attention using context
        4) residual connection and layer normalization
        5) feed forward layer
        6) residual layer and layer normalization
        7) return relu of tensor

        NOTES: This article may be of great use:
        https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """

        # inputs is the caption embedding, context sequence is image embedding
        # print('shape of inputs shape(image embedding)', inputs.shape)
        # print('shape of context seq(caption embedding)', context_sequence.shape)
        # inputs = inputs[:, :, :length]
        # k, v, q
        encoder_self_attention = self.self_atten(inputs, inputs, inputs)
        # print('encoder self attn shape', encoder_self_attention.shape)
        # print('inputs shape', inputs.shape)
        masked_encoder_self_attention = self.layer_norm(encoder_self_attention + inputs)

        if len(context_sequence.shape) != 3:
            context_sequence = tf.expand_dims(context_sequence, axis=1)
        #k, v, q = context_sequence, context_sequence, masked_encoder_self_attention
        unmasked_attention = self.cross_atten(context_sequence, context_sequence, masked_encoder_self_attention)
        unmasked_attention = self.layer_norm(unmasked_attention + masked_encoder_self_attention)

        ff = self.ff_layer(unmasked_attention)
        ff = self.layer_norm(ff + unmasked_attention)
        return tf.nn.relu(ff)

def positional_encoding(length, depth):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    depth = depth / 2
    ## Generate a range of positions and depths
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    ## Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    ## This serves as offset for the Positional Encoding
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Embed labels and apply positional offsetting
    """

    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size

        ## TODO: Implement Components

        ## Embed labels into an optimizable embedding space
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size, mask_zero=True)

        ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies.
        ## HINT: May want to use the function above...
        self.pos_encoding = positional_encoding(length=window_size, depth=embed_size)

    def call(self, x):
        ## TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.
        embdding = self.embedding(x)
        embdding *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        pos_code = self.pos_encoding
        # print('In positional encoding: embdding shape', embdding.shape)
        # print('In positional encoding: pos_code shape', pos_code.shape)
        return embdding + pos_code[:embdding.shape[1], :]

