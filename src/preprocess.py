"""
Do the preprocessing of the data, including: chop the ingredients into words.
Important:
1. 'data.p' is the raw data, which is a dictionary of dish name and ingredients list. Do NOT overwrite this file.

2. 'data_lemmatized.p' is the data after tokenization and lemmatization by spacy.
   After tokenizing and lemmatizing the FULL SIZE data, do NOT overwrite this file.

3. 'prep_data.p' collects information from 'data_lemmatized.p', including
    'X': X(dish names),
    'Y': Y(ingradient names),
    'dish_word2idx': dish_word2idx,
    'dish_idx2word': dish_idx2word,
    'dish_vocab_size': dish_vocab_size,
    'ingredient_word2idx': ingredient_word2idx,
    'ingredient_idx2word': ingredient_idx2word,
    'ingredient_vocab_size': ingredient_vocab_size,
    'window_size': window_size

4. I use words "tokenization" and "lemmatization" interchangeably.
"""



import csv
import re
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split 


def truncate(arr, size):
    if len(arr) > size:
        return arr[:size + 1]
    else:
        copy = arr.copy()
        copy += (size + 1 - len(arr)) * ['<pad>']
        return copy


def preprocess_sentence_list(sentence_list, window_size):
    for i, sentence in enumerate(sentence_list):
        sentence_new = ['<start>'] + sentence[:window_size - 1] + ['<end>']
        sentence_list[i] = truncate(sentence_new, window_size)



def preprocess_data_to_X_Y_for_tokenization(data):
    '''

    :parameter data: a dictionary of dish name and ingredients, loaded from data.p
    the structure looks like the raw_data above
    :return: X_train, Y_train, pairwisely
    X_train is a list of dish sentence(with white space and punctuations), 1 dimension(need spacy to do further tokenization)
    Y_train is a list of list of ingredient-sentence, 2 dimension(need spacy to do further tokenization for each ingredient-sentence)
    '''
    X = []
    Y = []
    for dish, ingredients in data.items():
        X.append(dish)
        Y.append(ingredients)
    return X, Y


def regex_tokenization_for_X_Y(X, Y, save_to_file=False):
    '''

    :param X: the preprocessed data X from the above function, should be a list of string(dish sentence)
    :param Y: the preprocessed data Y from the above function, should be a list of list of string(ingredient-sentence)
    split the dish sentence and ingredient sentence by space, remove non-alphanumeric characters.

    Pro: Fast
    Con: vocab size may be too large, vocab distribution may be sparse.

    :return: tokenized X and Y
    '''

    non_alpha_num = re.compile('[^a-zA-Z0-9 ]')

    for i, (x, y) in enumerate(zip(X, Y)):
        x = x.lower()
        x = re.sub(non_alpha_num, '', x)
        x_tokens = re.split('\s+', x)
        X[i] = x_tokens

        for j, ingredient in enumerate(y):
            ingredient = ingredient.lower()
            ingredient = re.sub(non_alpha_num, '', ingredient)
            ingredient_tokens = re.split('\s+', ingredient)
            ingredient = ' '.join(ingredient_tokens)
            y[j] = ingredient

    data_to_dump = {'X': X, 'Y': Y}

    if save_to_file:
        with open('data_lemmatized.p', 'wb') as f:
            pickle.dump(data_to_dump, f)

    return X, Y


def spacy_tokenization_for_X_Y(X, Y, save_to_file=False):
    '''

    :param X: the preprocessed data X from the above function, should be a list of string(dish sentence)
    :param Y: the preprocessed data Y from the above function, should be a list of list of string(ingredient-sentence)
    Need to do tokenization for each sentence, and turn it to a lemmatized sentence string:
    i.e. 'toasted sliced almonds' -> [toast, slice, almond] (after tokenization and remove punctuation)
    [toast, slice, almond] -> 'toast slice almond' (we treat each ingredient as a word)
    Hopefully, the word embedding for the two "words": 'olive oil' and 'extra virgin olive oil'
    are similar after training.

    Pro: make vocabulary size less and less sparse.
    Con: some ingredient name is not in correct English: ex. we say "sliced almonds" but not "slice almond".


    Some detail: spacy tokenization is sensitive to capitalization.
    for example, when we do nlp('Glazed Finger Wings'), it will output lemmas ['glaze', 'Finger', 'Wings']
    notice that the s in 'wings' is not truncated, maybe spacy treat it as a special noun.
    If we first lower() the original string, then the tokenizier may treat every word as a regular noun.
    But dish name shoud be somehow special, so, just do lower() AFTER tokenization.


    :return: lemmatized X and Y
    '''

    import spacy
    nlp = spacy.load('en_core_web_sm')
    total = len(X)
    for i, (x, y) in enumerate(zip(X, Y)):
        print(f'\r {i + 1}/{total} spacy lemmatization', end='')
        x_tokens = nlp(x)
        x_lemmatized = []
        for token in x_tokens:
            if not token.is_punct:
                x_lemmatized.append(token.lemma_.lower())
        X[i] = x_lemmatized

        for j, ingredient in enumerate(y):
            ingredient_tokens = nlp(ingredient)
            ingredient_lemmatized = []
            for token in ingredient_tokens:
                if not token.is_punct:
                    ingredient_lemmatized.append(token.lemma_.lower())
            ingredient = ' '.join(ingredient_lemmatized)
            y[j] = ingredient
    print()
    data_to_dump = {'X': X, 'Y': Y}

    if save_to_file:
        with open('data_lemmatized.p', 'wb') as f:
            pickle.dump(data_to_dump, f)

    return X, Y


def preprocess_paired_data(data=None, window_size=20, save_to_file=False, file_name='prep_data.p', data_size=None, min_frequency=50):
    if data is None:
        X = [['Glazed', 'Finger', 'Wings'],
                   ['Country', 'Scalloped', 'Potatoes', '&amp;', 'Ham', '(Crock', 'Pot)'],
                   ['Fruit', 'Dream', 'Cookies'], ['Tropical', 'Breakfast', 'Risotti'],
                   ['Linguine', 'W/', 'Olive,', 'Anchovy', 'and', 'Tuna', 'Sauce']]
        Y = [['chicken-wings', 'sugar,', 'cornstarch', 'salt', 'ground ginger', 'pepper', 'water', 'lemon juice',
                    'soy sauce'],
                   ['potatoes', 'onion', 'cooked ham', 'country gravy mix', 'cream of mushroom soup', 'water',
                    'cheddar cheese'],
                   ['butter', 'shortening', 'granulated sugar', 'eggs', 'baking soda', 'baking powder', 'vanilla',
                    'all-purpose flour', 'white chocolate chips', 'orange drink mix', 'colored crystal sugar'],
                   ['water', 'instant brown rice', 'pineapple tidbits', 'skim evaporated milk', 'raisins',
                    'sweetened flaked coconut', 'toasted sliced almonds', 'banana'],
                   ['anchovy fillets', 'tuna packed in oil', 'kalamata olive', 'garlic cloves', 'fresh parsley',
                    'fresh lemon juice', 'salt %26 pepper', 'olive oil', 'linguine']]
    else:
        X = data[0]
        Y = data[1]

    if data_size is not None:
        X = X[:data_size]
        Y = Y[:data_size]

    ingredient_word2idx = {}
    dish_word2idx = {}
    ingredient_vocab_size = 0
    dish_vocab_size = 0

    preprocess_sentence_list(X, window_size)
    preprocess_sentence_list(Y, window_size)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size = 0.99,random_state=42)

    ## preprocess test set
    # preprocess_sentence_list(X_test, window_size)
    # preprocess_sentence_list(Y_test, window_size)

    src_word_count = Counter()
    tgt_word_count = Counter()

    for sentence in X_train:
        src_word_count.update(sentence)
    for sentence in Y_train:
        tgt_word_count.update(sentence)


    def unk_process(sentence_list, counter, min_frequency):
        unk = 0
        for i, sentence in enumerate(sentence_list):
            for j, word in enumerate(sentence):
                if counter[word] <= min_frequency:
                    sentence[j] = '<unk>'
                    unk += counter[word]
        return unk

    unk_src = unk_process(X_train, src_word_count, min_frequency)
    unk_tgt = unk_process(Y_train, tgt_word_count, min_frequency)

    print(f'unk_src: {unk_src}, unk_tgt: {unk_tgt}')
    print(f'total src words: {sum(src_word_count.values())}, total tgt words: {sum(tgt_word_count.values())}')

    for x, y in zip(X_train, Y_train):
        for word in x:
            if word not in dish_word2idx:
                dish_word2idx[word] = dish_vocab_size
                dish_vocab_size += 1
        for word in y:
            if word not in ingredient_word2idx:
                ingredient_word2idx[word] = ingredient_vocab_size
                ingredient_vocab_size += 1

    dish_idx2word = {i: w for w, i in dish_word2idx.items()}
    ingredient_idx2word = {i: w for w, i in ingredient_word2idx.items()}

    for i, (x, y) in enumerate(zip(X_train, Y_train)):
        X_train[i] = [dish_word2idx[word] for word in x]
        Y_train[i] = [ingredient_word2idx[word] for word in y]

    data_to_dump = {'X': X_train,
                    'Y': Y_train,
                    'X_test': X_test,
                    'Y_test': Y_test,
                    'dish_word2idx': dish_word2idx,
                    'dish_idx2word': dish_idx2word,
                    'dish_vocab_size': dish_vocab_size,
                    'ingredient_word2idx': ingredient_word2idx,
                    'ingredient_idx2word': ingredient_idx2word,
                    'ingredient_vocab_size': ingredient_vocab_size,
                    'window_size': window_size
                    }


    if save_to_file:
        with open(file_name, 'wb') as processed_file:
            pickle.dump(data_to_dump, processed_file)

    return data_to_dump


def pipeline_for_tokenization(data, use_spacy=False, save_to_file=False):
    '''
    This function is integerated with all the above functions.

    :param data:  data.p, or same smaller size data with same structure as data.p
    :param spacy: if True, use spacy to do tokenization, else use regex
    :return: data with word2idx and idx2word, ready for tensorization
    '''

    # pairwisely arrange the data
    X, Y = preprocess_data_to_X_Y_for_tokenization(data)

    # tokenize the data
    lemmatizer = spacy_tokenization_for_X_Y if use_spacy else regex_tokenization_for_X_Y
    X, Y = lemmatizer(X, Y, save_to_file=save_to_file)

    # makeup data into tensor format
    prep_data = preprocess_paired_data(data=(X, Y), save_to_file=save_to_file)

    return prep_data



def get_data_p():
    '''
    Load data.p as a dict for further processing
    :return:
    '''
    with open('data.p', 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':

    with open('data_lemmatized2.p', 'rb') as f:
        d = pickle.load(f)


    # X = data['X']
    # Y = data['Y']
    #
    # d = preprocess_paired_data(data=(X, Y), save_to_file=True)
    
    
    X = d['X']
    Y = d['Y']

    data = preprocess_paired_data(data=(X, Y), save_to_file=True, file_name='prep_data_train_test_split.p',
                                  data_size=None, min_frequency=10) # restore this line later!!!

    print(data.keys())
    #
    # dish_idx2word = d['dish_idx2word']
    # ingredient_idx2word = d['ingredient_idx2word']
    # print(d['dish_vocab_size'], d['ingredient_vocab_size'])
    # for x, y in zip(X, Y):
    #     print(x, '->', y)
    #     for i in x:
    #         print(dish_idx2word[i], end=' ')
    #     print()
    #     for i in y:
    #         print(ingredient_idx2word[i], end=' ')
    #     print()
    #     print()



    # pipeline_for_tokenization(get_data_p(), use_spacy=True, save_to_file=True)
    # raw_data = {
    #     'Glazed Finger Wings': ['chicken-wings', 'sugar', 'cornstarch', 'salt', 'ground ginger', 'pepper', 'water',
    #                             'lemon juice', 'soy sauce'],
    #     'Country Scalloped Potatoes &amp; Ham (Crock Pot)': ['potatoes', 'onion', 'cooked ham', 'country gravy mix',
    #                                                          'cream of mushroom soup', 'water', 'cheddar cheese'],
    #     'Fruit Dream Cookies': ['butter', 'shortening', 'granulated sugar', 'eggs', 'baking soda', 'baking powder',
    #                             'vanilla', 'all-purpose flour', 'white chocolate chips', 'orange drink mix',
    #                             'colored crystal sugar'],
    #     'Tropical Breakfast Risotti': ['water', 'instant brown rice', 'pineapple tidbits', 'skim evaporated milk',
    #                                    'raisins', 'sweetened flaked coconut', 'toasted sliced almonds', 'banana'],
    #     'Linguine W/ Olive, Anchovy and Tuna Sauce': ['anchovy fillets', 'tuna packed in oil', 'kalamata olive',
    #                                                   'garlic cloves', 'fresh parsley', 'fresh lemon juice',
    #                                                   'salt %26 pepper', 'olive oil', 'linguine']}
    #
    # use_data_p = False
    #
    # data = get_data_p() if use_data_p else raw_data
    # X, Y = preprocess_data_to_X_Y_for_tokenization(data)
    #
    # print('before tokenization:')
    # for x, y in zip(X, Y):
    #     print(f'\'{x}\':', y)
    #
    # # X, Y = spacy_tokenization_for_X_Y(X, Y)
    # X, Y = regex_tokenization_for_X_Y(X, Y, save_to_file=False)
    #
    # print('after tokenization:')
    # for x, y in zip(X, Y):
    #     print(f'\'{x}\':', y)
    #
    # prep_data = preprocess_paired_data(data=(X, Y), window_size=20, save_to_file=False)
    # X, Y = prep_data['X'], prep_data['Y']
    # print('after preprocessing:')
    # for x, y in zip(X, Y):
    #     print(f'\'{x}\':', y)
    # X, Y = np.array(X), np.array(Y)
    # print(X.shape, Y.shape)




