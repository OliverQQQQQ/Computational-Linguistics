#!/usr/bin/env python
import argparse
from util import load_function_words, parse_federalist_papers, labels_to_key, labels_to_y, split_data, \
                 find_zero_rule_class, apply_zero_rule, calculate_accuracy
import numpy as np
from nltk import word_tokenize
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
import random
random.seed(111)

# TODO create a function that loads all the essays into a matrix
# load the attributed essays into a feature matrix
def load_features(reviews, feature_key):
    """"
    :return: numpy matrix
    """
    feature_np = np.zeros(((len(reviews)), len(feature_key)), dtype=np.int)
    for i, review in enumerate(reviews):
        review_words = word_tokenize(review.lower())
        for j, word in enumerate(feature_key):
            these_words = [w for w in review_words if w == word]
            feature_np[i, j] = len(these_words)
    return feature_np


def main(data_file, vocab_path):
    """Build and evaluate Naive Bayes classifiers for the federalist papers"""
    authors, essays, essay_ids = parse_federalist_papers(data_file)
    function_words = load_function_words(vocab_path)
    # load the attributed essays into a feature matrix
    X = load_features(essays, function_words)


    # TODO: load the author names into a vector y, mapped to 0 and 1, using functions from util.py
    labels_map = labels_to_key(authors)
    y = np.asarray(labels_to_y(authors, labels_map))
    print(f"X has shape {X.shape} and dtype {X.dtype}")
    print(f"y has shape {y.shape} and dtype {y.dtype}")


    # TODO shuffle, then split the data
    train, test = split_data(X, y, 0.25)
    data_size_after = len(train[1]) + len(test[1])

    assert data_size_after == y.size, f"Number of datapoints after split {data_size_after} must match size before {y.size}"
    print(f"{len(train[0])} in train; {len(test[0])} in test")


    # TODO: train a multinomial NB model, evaluate on validation split
    nb_mul = MultinomialNB()
    nb_mul.fit(train[0], train[1])

    pred_mul = nb_mul.predict(test[0])
    acc_mul = metrics.accuracy_score(test[1],pred_mul)
    print(f"Accuracy of Multinomial NB method: {acc_mul:0.03f}")


    # TODO: train a Bernoulli NB model, evaluate on validation split
    nb_ber = BernoulliNB()
    nb_ber.fit(train[0], train[1])

    pred_ber = nb_ber.predict(test[0])
    acc_ber = metrics.accuracy_score(test[1], pred_ber)
    print(f"Accuracy of Bernoulli NB method: {acc_ber:0.03f}")


    # TODO: fit the zero rule
    # learn zero rule on train
    most_frequent_class = find_zero_rule_class(train[1])

    # apply zero rule to test reviews
    test_predictions = apply_zero_rule(test[0], most_frequent_class)

    # score accuracy
    test_accuracy = calculate_accuracy(test_predictions, test[1])
    print(f"Accuracy of Zero rule: {test_accuracy:0.03f}")

    # lookup label string from class #
    author_key = labels_to_key(authors)
    reverse_author_key = {v: k for k, v in author_key.items()}
    print(f"The author predicted by the Zero rule is {reverse_author_key[most_frequent_class]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="federalist_dev.json",
                        help='path to author dataset')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.function_words_path)
