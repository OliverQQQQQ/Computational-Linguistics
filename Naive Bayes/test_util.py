from util import split_data, labels_to_key, parse_federalist_papers, labels_to_y, find_zero_rule_class, \
    apply_zero_rule, calculate_accuracy
import argparse

def main(data_file):
    print(data_file)

    # load the data
    authors, essays, essay_ids = parse_federalist_papers(data_file)
    num_essays = len(essays)
    print(f"Working with {num_essays} reviews")

    # create a key that links author id string -> integer
    author_key = labels_to_key(authors)
    print(len(author_key))
    print(author_key)

    # convert all the labels using the key
    y = labels_to_y(authors, author_key)
    assert y.size == len(authors), f"Size of label array (y.size) must equal number of labels {len(authors)}"

    # shuffle and split the data
    train, test = split_data(essays, y, 0.3)
    data_size_after = len(train[1]) + len(test[1])

    assert data_size_after == y.size, f"Number of datapoints after split {data_size_after} must match size before {y.size}"
    print(f"{len(train[0])} in train; {len(test[0])} in test")

    # learn zero rule on train
    train_y = train[1]
    most_frequent_class = find_zero_rule_class(train_y)

    # lookup label string from class #
    reverse_author_key = {v:k for k,v in author_key.items()}
    print(f"The most frequent class is {reverse_author_key[most_frequent_class]}")

    # apply zero rule to test reviews
    test_predictions = apply_zero_rule(test[0], most_frequent_class)
    print(f"Zero rule predictions on held-out data: {test_predictions}")

    # score accuracy
    test_y = test[1]
    test_accuracy = calculate_accuracy(test_predictions, test_y)
    print(f"Accuracy of zero rule: {test_accuracy:0.03f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test supervised learning utilities')
    parser.add_argument('--path', type=str, default="federalist_dev.json",
                        help='path to author dataset')
    args = parser.parse_args()

    main(args.path)