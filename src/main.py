import re
import pickle

import argparse
import numpy as np
import pickle
import tensorflow as tf
from model import accuracy_function, similarity_function, jaccard_similarity, loss_function, DishIngredientPredictorModel
from rnn import RNN
from transformer import Transformer

def compile_model(model):
    '''Compiles model'''
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=loss_function,
                  metrics=[accuracy_function, similarity_function, jaccard_similarity])


def train_model(model, src_inputs, tgt_inputs, src_pad_idx, tgt_pad_idx, args, valid):
    '''Trains model and returns model statistics'''
    stats = []
    # print('train_model is called!')
    try:
        for epoch in range(args.epochs):
            stats += [model.train(tgt_inputs, src_inputs, src_pad_idx, tgt_pad_idx, batch_size=args.batch_size)]
            if args.check_valid:
                model.test(valid[0], valid[1], src_pad_idx, tgt_pad_idx, batch_size=args.batch_size)
    except KeyboardInterrupt as e:
        if epoch > 0:
            print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
        else:
            raise e

    return stats


def build_model(args):
    with open('prep_data.p', 'rb') as f:
        data = pickle.load(f)

    train_size = -10000
    src_train_inputs = data['X'][:-train_size]
    tgt_train_inputs = data['Y'][:-train_size]
    src_test_inputs = data['X'][-1000:]
    tgt_test_inputs = data['Y'][-1000:]
    src_w2i = data['dish_word2idx']
    tgt_w2i = data['ingredient_word2idx']
    src_i2w = data['dish_idx2word']
    tgt_i2w = data['ingredient_idx2word']
    src_vocab_size = data['dish_vocab_size']
    tgt_vocab_size = data['ingredient_vocab_size']

    print('src_vocab_size', src_vocab_size)
    print('tgt_vocab_size', tgt_vocab_size)
    print(src_w2i)

    dt = tf.int32
    src_train_inputs = tf.convert_to_tensor(src_train_inputs, dtype=dt)
    tgt_train_inputs = tf.convert_to_tensor(tgt_train_inputs, dtype=dt)
    src_test_inputs = tf.convert_to_tensor(src_test_inputs, dtype=dt)
    tgt_test_inputs = tf.convert_to_tensor(tgt_test_inputs, dtype=dt)

    print('src_train_inputs shape: ', src_train_inputs.shape)
    print('tgt_train_inputs shape: ', tgt_train_inputs.shape)
    print('src_test_inputs dtype: ', src_test_inputs.shape)
    print('tgt_test_inputs dtype: ', tgt_test_inputs.shape)

    predictor_class ={
        'rnn': RNN,
        'transformer': Transformer
    }[args.type]

    hidden_size = args.hidden_size
    window_size = args.window_size

    predictor = predictor_class(src_vocab_size, tgt_vocab_size, hidden_size, window_size, nhead=args.nhead)

    model = DishIngredientPredictorModel(
        predictor,
        src_w2i,
        src_i2w,
        tgt_w2i,
        tgt_i2w
    )

    compile_model(model)

    if True:
        train_model(model, src_train_inputs, tgt_train_inputs, src_w2i['<pad>'], tgt_w2i['<pad>'], args, (src_test_inputs, tgt_test_inputs))

    if 'X_test' in data:
        X_test = data['X_test']
        Y_test = data['Y_test']

        try:
            for dish, ingredients in zip(X_test, Y_test):
                print('dish', dish)
                print('ingredients', ingredients)
                prediction = model.predict(dish)
                print('predicted ingredients', prediction)
                inter = set(ingredients).intersection(set(prediction))
                union = set(ingredients).union(set(prediction))
                print('Jaccard Similarity:', len(inter) / len(union))
                print()
        except:
            print('Error happens in batch test.')

    return model, (src_test_inputs, tgt_test_inputs)


def parse_args(args=None):
    """
    Perform command-line argument parsing (other otherwise parse arguments with defaults).
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example:
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Let's train some neural nets!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--type',           required=True,              choices=['rnn', 'transformer'],     help='Type of model to train')
    parser.add_argument('--task',           required=False,              choices=['train', 'test', 'both'],  help='Task to run')
    parser.add_argument('--data',           required=False,              help='File path to the assignment data file.')
    parser.add_argument('--epochs',         type=int,   default=3,      help='Number of epochs used in training.')
    parser.add_argument('--lr',             type=float, default=1e-3,   help='Model\'s learning rate')
    parser.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Model\'s optimizer')
    parser.add_argument('--nhead',          type=int,   default=3,      help='Number of heads in the transformer model')
    parser.add_argument('--batch_size',     type=int,   default=100,    help='Model\'s batch size.')
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--window_size',    type=int,   default=20,     help='Window size of text entries.')
    parser.add_argument('--chkpt_path',     default='',                 help='where the model checkpoint is')
    parser.add_argument('--check_valid',    default=False,               action="store_true",  help='if training, also print validation after each epoch')
    parser.add_argument('--debug',          default=False,               action="store_true",  help='skip training')
    if args is None:
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

if __name__ == '__main__':

    model = build_model(parse_args('--type transformer --nhead 3 --epochs 1 --batch_size 100'.split()))
    model = model[0]


    print('---------Manual Test----------')
    dishes = [
        ['fried', 'rice'],
        ['pepper', 'steak'],
        ['chicken', 'soup'],
        ['butter', 'pork'],
        ['orange','chicken'],
        ['apple','chicken'],
        ['veggie','dumplings'],
        ['chocolate','cookies'],
        ['meatloaf']
    ]
    for dish in dishes:
        print(dish)
        print(model.predict(dish))
        print()

