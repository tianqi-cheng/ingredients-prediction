import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_text as tf_text
import spacy
from scipy import sparse
from collections import Counter

if __name__ == '__main__':
    # a = ['water', 'grit', 'salt', 'cheddar cheese', 'garlic', 'olive oil']
    # s = 'I love you but he loves her and my friends are all nlp big fans.'
    #
    # counter = Counter()
    # nlp = spacy.load('en_core_web_sm')
    # doc = nlp(s)
    # counter.update(doc)
    # print(counter)
    a = [
        [1, 3, 2],
        [5, 6, 4],
        [12, 11, 10]
    ]

    b = [
        [0.5, 0.3, 0.2],
        [0.1, 0.2, 0.7],
        [0.4, 0.2, 0.3]
    ]

    c = [
        [1, 3, 2],
        [0, 0, 0],
        [0, 0, 0]
    ]

    d = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    a = tf.convert_to_tensor(a, dtype=tf.float32)
    a = tf.expand_dims(a, axis=0)
    b = tf.convert_to_tensor(b, dtype=tf.float32)
    b = tf.expand_dims(b, axis=0)
    c = tf.convert_to_tensor(c, dtype=tf.float32)
    labels = tf.expand_dims(c, axis=0)
    d = tf.convert_to_tensor(d, dtype=tf.float32)
    preds = tf.expand_dims(d, axis=0)
    cos_sim = tf.keras.losses.CosineSimilarity()

    a = tf.convert_to_tensor([1, 2, 3, 5])
    b = tf.convert_to_tensor([1, 2, 4, 6])

    a = set(a.numpy().tolist())
    b = set(b.numpy().tolist())
    print(len(a | b))
    print(len(a & b))

