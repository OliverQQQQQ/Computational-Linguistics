import numpy as np
from collections import Counter
import json

def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words

# TODO: lab 1
def parse_federalist_papers(data_file):
    authors = []
    texts = []
    essay_ids = []
    with open(data_file, 'r') as df:
        data = json.load(df)  # load as list
    for item in data:
        author, essay, text = item
        authors.append(author)
        texts.append(text)
        essay_ids.append(essay)
    return authors, texts, essay_ids

# TODO: write this function (lab)
def shuffle_dataset(data, id_strs):
    """
    Shuffles a list of datapoints and their id's in unison
    :param data: iterable, each item a datapoint
    :param id_strs: iterable, each item an id
    :return: tuple (shuffled_data, shuffled_id_strs)
    """
    new_order = np.random.permutation(len(id_strs))
    # flexible to work with lists (i.e. of strings) or np
    if isinstance(id_strs, np.ndarray):
        shuffled_ids = id_strs[new_order]
    else:
        shuffled_ids = [id_strs[i] for i in new_order]
    if isinstance(data, np.ndarray):
        shuffled_data = data[new_order]
    else:
        shuffled_data = [data[i] for i in new_order]
    return (shuffled_data, shuffled_ids)

# TODO: write this function (lab1, homework)
def split_data(X, file_ids, test_percent, shuffle=True):
    """
    Splits dataset for supervised learning and evaluation
    :param X: iterable of features
    :param file_ids: iterable of file id's corresponding the features in X
    :param test_percent: percent data to
    :param shuffle:
    :return: two tuples, (X_train, file_ids_train), (X_test, file_ids_test)
    """
    if shuffle:
        X, file_ids = shuffle_dataset(X, file_ids)
    data_size = len(X)
    num_test = int(test_percent * data_size)

    train = (X[:-num_test], file_ids[:-num_test])
    test = (X[-num_test:], file_ids[-num_test:])
    return train, test

# TODO: write this function (lab1, homework)
def labels_to_key(labels):
    """
    Creates a mapping from string representations of labels to integers
    :param labels:
    :return: label_key, dict {str: int}
    """
    label_set = set(labels)
    label_key = {}
    for i, label in enumerate(label_set):
        label_key[label] = i
    return label_key

# TODO: write this function (lab1, homework)
def labels_to_y(labels, label_key):
    """
    :param labels: list of strings
    :param label_key: dictionary {str: int}
    :return: numpy vector y
    """
    y = np.zeros(len(labels), dtype=np.int)
    for i,l in enumerate(labels):
        y[i] = label_key[l]
    return y

# TODO: write this function (lab1, homework)
def find_zero_rule_class(train_y):
    """
    Determines the class predicted by the zero rule algorithm
    :param train_y: training labels
    :return: most_freq, the most frequent element in train_y
    """
    class_counts = Counter(train_y)
    #print(class_counts)
    most_freq = max(class_counts, key=lambda k: class_counts[k])
    return most_freq

# TODO: write this function (lab1, homework)
def apply_zero_rule(X, zero_class):
    """
    Predicts most frequent class using zero rule algorithm
    :param X: iterable, data to classify
    :param zero_class: class to predict
    :return: classifications: numpy array
    """
    classifications = np.zeros(len(X), dtype=np.int)
    # assign every y the zero class
    classifications[:] = zero_class
    return classifications

# TODO: write this function (lab1, homework)
def calculate_accuracy(predicted, gold):
    """
    :param predicted: iterable, system output
    :param gold: iterable, gold standard labels
    :return: accuracy: float in range [0,1]
    """
    predicted = np.asarray(predicted)
    gold = np.asarray(gold)
    correct = (predicted == gold).sum()
    return correct / gold.size

