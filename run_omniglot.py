"""
Train a model on Omniglot.
"""

import random

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import OmniglotModel
from supervised_reptile.omniglot import read_dataset, split_dataset, augment_dataset
from supervised_reptile.train import train

DATA_DIR = 'data/omniglot.nosync'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    train_set = list(augment_dataset(train_set))
    test_set = list(test_set)

    model = OmniglotModel(args.classes, **model_kwargs(args))

    with tf.Session() as sess:
        resume_itr = 0 # Zero iterations have already been trained.

        if args.pretrained or args.test: # It must be pretrained to test it.
            print('Restoring from checkpoint...')
            saved_model = tf.train.latest_checkpoint(args.checkpoint)
            tf.train.Saver().restore(sess, saved_model)

            resume_itr = int(saved_model[saved_model.index('model.ckpt') + 11:])

        if not args.test:
            print('Training...')
            train(sess, model, train_set, test_set, args.checkpoint, resume_itr, **train_kwargs(args))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

if __name__ == '__main__':
    main()
