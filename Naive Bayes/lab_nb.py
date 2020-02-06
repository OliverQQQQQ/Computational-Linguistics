#!/usr/bin/env python
import argparse
from util import parse_federalist_papers, load_function_words
from nltk import word_tokenize


def word_probabilities(list_of_reviews, feature_list):
    """calculates probabilities of each feature given this dataset using Laplace smoothing
    returns a dict {feature_1: probability_1, ... feature_n: probability_n}"""
    all_review_tokens = []
    for review in list_of_reviews:
        #all_review_tokens.extend(review.strip().lower().split())
        all_review_tokens.extend(word_tokenize(review.lower()))
    author_counts = {}
    for feature in feature_list:
        count = len([t for t in all_review_tokens if t == feature])
        author_counts[feature] = count + 1 # add smoothing
    total_count = sum(author_counts.values())
    author_probs = {f: (count / total_count) for f, count in author_counts.items()}
    return author_probs


def score(review, author_prob, feature_probs):
    """Calculates a naive bayes score for a string, given class estimate and feature estimates"""
    tokenized_review = word_tokenize(review.lower())
    p = author_prob
    for feature, prob in feature_probs.items():
        feature_count = len([w for w in tokenized_review if w == feature])
        for i in range(feature_count):
            p = p * prob
    return p

def main(data_file, features):
    """extract function word features from a text file"""

    # TODO: create a dictionary from author -> list of essays for two authors we will model
    authors, essays, essay_ids = parse_federalist_papers(data_file)
    essay_dict = {"HAMILTON":[], "MADISON":[]}
    for author, essay in zip(authors, essays):
        if author in essay_dict:
            essay_dict[author].append(essay)



    # hold out one review per author to test the model
    training_essays = {author: essays[:-1] for author, essays in essay_dict.items()}
    heldout_essays = {author: essays[-1] for author, essays in essay_dict.items()}


    # TODO estimate author probabilities. Creates a dict {author_1: probability_1, ...}
    total_essays = sum([len(essay_list) for essay_list in training_essays.values()])
    author_probs = {}


    for author in training_essays:
        author_probs[author] = len(training_essays[author]) / total_essays
    print(f"Author prior: {author_probs}")

    author_word_probs = {}
    for author in training_essays:
        word_probs = word_probabilities(training_essays[author], features)
        author_word_probs[author] = word_probs
    print(author_word_probs)


    for author in heldout_essays:
        essay = heldout_essays[author]
        essay_snippet = " ".join(essay[500:700].split()) # print a snippet
        print(f"\nChecking heldout essay by {author}")
        print(f"{essay_snippet}")

        for testauthor in heldout_essays:
            this_author_probability = author_probs[testauthor]
            word_probs = author_word_probs[testauthor]
            prob = score(essay, this_author_probability, word_probs)
            print(f"model probability essay is from {testauthor}: {prob:0.02}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='artisinal naive bayes lab')
    parser.add_argument('--path', type=str, default="federalist_dev.json",
                        help='path to author data')

    args = parser.parse_args()
    features = ["in", "while", "until", "which", "how"]
    #features = load_function_words("ewl_function_words.txt")

    main(args.path, features)
