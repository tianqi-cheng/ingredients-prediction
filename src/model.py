import numpy as np
import tensorflow as tf

import transformer
from preprocess import truncate


class DishIngredientPredictorModel(tf.keras.Model):

    def __init__(self, predictor, src_w2i, src_i2w, tgt_w2i, tgt_i2w, **kwargs):
        super().__init__(**kwargs)
        self.predictor = predictor
        self.src_w2i = src_w2i
        self.src_i2w = src_i2w
        self.tgt_w2i = tgt_w2i
        self.tgt_i2w = tgt_i2w


    @tf.function
    def call(self, dish_names, ingredient_names, src_padding_mask=None, tgt_padding_mask=None):
        print('dish_names shape', dish_names.shape)
        print('ingredient_names shape', ingredient_names.shape)
        print('predictor type', type(self.predictor))
        return self.predictor(dish_names, ingredient_names, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)

    def predict(self, dish_names):
        if type(self.predictor) == transformer.Transformer:
            dish_names = truncate(dish_names, self.predictor.window_size - 1)

        w2i = lambda dish: self.src_w2i[dish] if dish in self.src_w2i else self.src_w2i['<unk>']
        tokens = [w2i(word) for word in dish_names]
        src_token = tf.convert_to_tensor(tokens)

        # to make src_token a 3D tensor [batch, window, embedding],
        # where here batch is 1, since we only predict one dish.
        src_token = tf.expand_dims(src_token, axis=0)

        tgt_token = self.predict_token(src_token)
        lst = []
        for sentence in tgt_token:
            for each in sentence:
                lst.append(self.tgt_i2w[tf.get_static_value(each)])

        return ', '.join([ingredient for ingredient in lst[1:] if ingredient not in ['<end>', '<pad>', '<start>', '<unk>']])

    def predict_token(self, src_tokens):
        num_tokens = src_tokens.shape[0]
        tgt_token = self.greedy_decode(src_tokens, max_len=20)
        return tgt_token

    def encode(self, src_tokens):
        # print('show predictor type', type(self.predictor))
        return self.predictor.encode(src_tokens)

    def decode(self, tgt_inputs, encoder_state):
        return self.predictor.decode(tgt_inputs, encoder_state)

    def greedy_decode(self, src_tokens, max_len, start_symbol='<start>', end_symbol='<end>'):

        if type(self.predictor) == transformer.Transformer:
            hidden_state = self.encode(src_tokens)
        else:
            hidden_output, hidden_state = self.encode(src_tokens)

        seen_ids = set()
        sentence = [self.tgt_w2i[start_symbol]]
        ys = tf.convert_to_tensor([sentence])
        for i in range(max_len):
            out = self.decode(ys, hidden_state)

            # next_candidates = tf.math.top_k(out[:, -1, :], k=20).indices.numpy().tolist()[0]
            # to_add = None
            # for next_word in next_candidates:
            #     if next_word not in seen_ids:
            #         seen_ids.add(next_word)
            #         to_add = next_word
            #         break
            # if to_add is None:
            #     break
            # singleton = tf.convert_to_tensor([to_add])
            # singleton = tf.expand_dims(singleton, 0)
            # ys = tf.concat([ys, singleton], axis=1)
            # if to_add == self.tgt_w2i[end_symbol]:
            #     break

            # Original code for greedy decoding, single word at a time, DO NOT REMOVE
            next_word = tf.math.argmax(out[:, -1], axis=1, output_type=tf.int32)
            ys = tf.concat([ys, tf.expand_dims(next_word, axis=1)], axis=1)
            if self.tgt_i2w[tf.get_static_value(next_word[0])] == end_symbol:
                break
        return ys

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss_function = loss
        self.accuracy_function = metrics[0]
        self.similarity_function = metrics[1]
        self.jaccard_similarity = metrics[2]

    def train(self, train_ingredients, train_dishes, src_padding_index, tgt_padding_index, batch_size=100):

        avg_loss = 0
        avg_acc = 0
        avg_prp = 0

        num_batches = max(1, int(len(train_ingredients) / batch_size))

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(train_ingredients)+1, batch_size)):
            start = end - batch_size
            batch_dishes = train_dishes[start:end, :-1]
            decoder_input = train_ingredients[start:end, :-1]
            decoder_labels = train_ingredients[start:end, 1:]
            src_padding_mask = tf.cast(tf.math.equal(batch_dishes, src_padding_index), tf.float32)
            tgt_padding_mask = tf.cast(tf.math.equal(decoder_input, tgt_padding_index), tf.float32)
            with tf.GradientTape() as tape:
                predictions = self.call(batch_dishes, decoder_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
                mask = decoder_labels != tgt_padding_index
                loss = self.loss_function(predictions, decoder_labels, mask)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            accuracy = self.accuracy_function(predictions, decoder_labels, mask)
            # similarity = self.similarity_function(predictions, decoder_labels, self)
            jaccard_similarity = self.jaccard_similarity(predictions, decoder_labels)

            total_loss += loss
            total_seen += num_predictions
            total_correct += accuracy

            avg_loss = total_loss / total_seen
            avg_acc = total_correct / total_seen
            avg_prp = np.exp(avg_loss)
            print(f'\rTrain {index+1}/{num_batches} - loss: {avg_loss:.4f} - jaccard_similarity: {jaccard_similarity:.4f} - perplexity: {avg_prp:.4f}', end='')

        print()

        return avg_loss, avg_acc, avg_prp


    def test(self, train_ingredients, train_dishes, src_padding_index, tgt_padding_index, batch_size=100):

        avg_loss = 0
        avg_acc = 0
        avg_prp = 0

        num_batches = max(1, int(len(train_ingredients) / batch_size))

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(train_ingredients)+1, batch_size)):
            start = end - batch_size
            batch_dishes = train_dishes[start:end, :-1]
            decoder_input = train_ingredients[start:end, :-1]
            decoder_labels = train_ingredients[start:end, 1:]
            src_padding_mask = tf.cast(tf.math.equal(batch_dishes, src_padding_index), tf.float32)
            tgt_padding_mask = tf.cast(tf.math.equal(decoder_input, tgt_padding_index), tf.float32)

            predictions = self.call(batch_dishes, decoder_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
            mask = decoder_labels != tgt_padding_index
            loss = self.loss_function(predictions, decoder_labels, mask)

            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            accuracy = self.accuracy_function(predictions, decoder_labels, mask)
            # similarity = self.similarity_function(predictions, decoder_labels, self)
            jaccard_similarity = self.jaccard_similarity(predictions, decoder_labels)

            total_loss += loss
            total_seen += num_predictions
            total_correct += accuracy

            avg_loss = total_loss / total_seen
            avg_acc = total_correct / total_seen
            avg_prp = np.exp(avg_loss)
            print(f'\rTest {index+1}/{num_batches} - loss: {avg_loss:.4f} - jaccard_similarity: {jaccard_similarity:.4f} - perplexity: {avg_prp:.4f}', end='')

        print()

        return avg_loss, avg_acc, avg_prp


def accuracy_function(prbs, labels, mask):
    correct_classes = tf.math.argmax(prbs, axis=-1, output_type=tf.int32) == labels
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
    return accuracy


def jaccard_similarity(prbs, labels):
    words_pred = tf.math.argmax(prbs, axis=-1, output_type=tf.int32)

    redundant = {'<unk>', '<start>', '<end>'}

    similarity = 0.0
    for s1, s2 in zip(words_pred, labels):
        words_unique = set(s1.numpy().tolist())
        label_unique = set(s2.numpy().tolist())
        words_unique = words_unique - redundant
        label_unique = label_unique - redundant

        similarity += len(words_unique & label_unique) / len(words_unique | label_unique)
    return similarity / len(words_pred)


def similarity_function(prbs, labels, model):
    words_pred = tf.math.argmax(prbs, axis=-1, output_type=tf.int32)

    words_pred_emb = model.predictor.get_embedding(words_pred)
    labels_emb = model.predictor.get_embedding(labels)

    words_pred_emb = tf.sort(words_pred_emb, axis=-1)
    labels_emb = tf.sort(labels_emb, axis=-1)
    # print('predction:', words_pred_emb)
    # print('decoder_labels:', labels_emb)

    sim = tf.keras.losses.CosineSimilarity()
    sz = tf.shape(words_pred_emb)[0] * tf.shape(words_pred_emb)[1] * tf.shape(words_pred_emb)[2]
    w = tf.reshape(words_pred_emb, sz)
    l = tf.reshape(labels_emb, sz)
    similarity = sim(w, l)
    return similarity


def loss_function(prbs, labels, mask):

    masked_labs = tf.boolean_mask(labels, mask)
    masked_prbs = tf.boolean_mask(prbs, mask)
    scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs, from_logits=True)
    loss = tf.reduce_sum(scce)
    return loss

