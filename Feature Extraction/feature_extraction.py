#!/usr/bin/env python
import argparse
import numpy as np
from util import load_function_words, load_reviews, split_data, feature_matrix

# do not alter this function
def check_splits(train, test, X, ids):
    """verify that all data is retained after splitting"""
    # check X
    train_X = train[0]
    val_X = test[0]
    sum_after = train_X.sum()+val_X.sum()
    assert sum_after == X.sum(), \
        f"Sum of features in Train+Test {sum_after} must equal sum of features before splitting {X.sum}"

    all_ids = train[1] + test[1]
    assert set(all_ids) == set(ids), "Set of ids in Train+Test must equal set of ids before splitting"
    # if we didn't crash, everything's good!
    print("Split checks passed!")



def main(data_file, vocab_path):
    """extract function word features from a text file"""

    ### load resources and text file
    function_words = load_function_words(vocab_path)
    reviews, ids = load_reviews(data_file)


    ### appropriately shape and fill this matrix
    review_features = np.zeros((len(reviews),len(function_words)), dtype=np.int)
    review_features = feature_matrix(reviews, function_words)
    # row is which review
    # column is which word
    print(f"Numpy array has shape {review_features.shape} and dtype {review_features.dtype}")


    ### Calculate these from review_features
    column_sum = np.sum(review_features,axis=0)
    most_common_count = max(column_sum)

    index = np.where(column_sum == column_sum.max())
    most_common_word = function_words[index[0][0]]

    print(f"Most common word: {most_common_word}, count: {most_common_count}")


    ### Find any features that weren't in the data (i.e. columns that sum to 0)
    index = np.where(column_sum == 0)
    zero_inds = index[0]
    if len(zero_inds)>0:
        print("No instances found for: ")
        for ind in zero_inds:
            print(f"  {function_words[ind]}")
    else:
        print("All function words found")

    matrix_sum = review_features.sum()
    print(f"Sum of raw count matrix: {matrix_sum}")


    ### make a binary feature vector from your count vector
    word_binary = np.copy(review_features)
    for i in range(len(reviews)):
        word_binary[i] = np.where(word_binary[i]>0, 1, 0)

    word_binary_sum = word_binary.sum()
    print(f"Sum of binary matrix: {word_binary_sum}")


    ### normalize features by review length (divide rows by number of words in the review)
    norm_reviews = np.copy(review_features)

    for i in range(len(reviews)):
        for j in range(len(function_words)):
            norm_reviews[i,j] = norm_reviews[i,j] / norm_reviews[i].sum()

    norm_reviews_sum = norm_reviews.sum()
    print(f"Sum of normed matrix: {norm_reviews_sum}")


    ### remove features from <review_features> that occur less than <min_count> times
    min_count = 100
    min_matrix = np.copy(review_features)

    index = np.where(column_sum <= min_count)
    mincnt_ind = index[0]

    functionword_min_matrix = []
    for i in range(len(function_words)):
        if i not in mincnt_ind:
            functionword_min_matrix.append(function_words[i])

    min_matrix = feature_matrix(reviews, functionword_min_matrix)

    min_matrix_shape = min_matrix.shape
    print(f"Shape after removing features that occur < {min_count} times: {min_matrix_shape}")


    ### split the dataset by updating the function above
    train, val = split_data(review_features, ids, 0.3)

    # Code below that all your data has been retained in your splits; do not edit.
    # Must all print True

    check_splits(train, val, review_features, ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="imdb_grade.txt",
                        help='path to input with one review per line')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    print(args.path)
    main(args.path, args.function_words_path)
