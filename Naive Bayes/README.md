# Naive Bayes for authorship attribution

# Experimental results
Data used for experiemental a number of federalist papers written by the author 'Hamilton' and 'Madison'. Purpose of this experiement is to predict the authorship using the text. There three methods are adopted for the prediction. Baseline experiment uses zero rule, Hamilton is predict by the zero rule since he has the most papers in this dataset. Besides that, Multinomial Naive Bayes and Bernoulli Naive Bayes are used as well. Count feature and binary feature are fed to the model repectively.

To measure and compare results of different methods, accuracy is used as the metric. Results showed that both Naive Bayes methods have very high accuracy, which are consitently being above 0.85. Zero rule methods yeilds a result range from 0.5 to 0.75, which performs worse than the other two NB methods.

# Files

## `federalist_dev.json` and `federalist_test.json`

json files containing the text of federalist papers written by Madison, Hamilton, or disputed between the two authors.
The dev file is all labeled. It can be split and used for development (training and validation). 
The test file contains only the disputed papers.

## Lab, week 1 : `util.py`

Implements utility functions for supervised learning to be imported in `lab_nb.py` and `multinomial_nb.py`.
There is no main routine, so a helper script `test_util.py` confirms that they work.

The functions are 
* splitting data (copy-paste from your last homework?)
* creating labels as numpy arrays
* implementing the zero-rule algorithm as a baseline / scoring accuracy

Usage: `python test_util.py --path federalist_dev.json`

## Lab, week 2 : `lab_nb.py`

This "artisinal" Naive Bayes lab implements the math of the model by hand!

Usage: `python lab_nb.py --path federalist_dev.json`

## Homework : `multinomial_nb.py`

Usage: `python multinomial_nb.py --function_words_path ewl_function_words.txt --path federalist_dev.json`

Apply `sklearn.naive_bayes.MultinomialNB` and `BernoulliNB` to two authors, as defined in the starter code. 
Consult the scikit learn docs to better understand how to interact with this model.

Refer to the feature extraction homework to create data in the right format for this model 
- i.e. concatenate all feature vectors to create input matrix X; create a label vector y.

Assign your data to train and test sets: 75% train and 25% test. use this same split for all experiments.


Fit and evaluate three models; our metric is accuracy:
* zero-rule baseline
* Multinomial Naive Bayes with count features
* Bernoulli Naive Bayes with binary features

Update this README with a brief summary (~2 paragraphs) of the dataset, methods and results (i.e. model accuracy), 
comparing the test results on your two models and the baseline. 
Include which author is predicted by the zero rule baseline.

_Naive Bayes is *deterministic*, meaning the model's probability estimates are always the same, given the same inputs. 
However, your random split may be different each time, depending on your implementation. 
You may see very different scores if you rerun your code._

