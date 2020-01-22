# Feature Extraction

Create and manipulate feature vectors from text using numpy.

It is possible to manipulate the matrices in this homework  using iteration, but numpy
provides more efficient methods - please use the numpy docs.

This homework practices:
* numpy arrays
* operations on arrays - entire array and only certain axes
* count vectors as features, binary vectors, normalized count vectors
* slicing arrays
* shuffling and splitting off a percent of random held-out data. Some good solutions for numpy shuffling:
https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
* documentation - please update comments and docstrings and add to this README.md with a brief description


# Files

## imdb_practice.txt / imdb_grade.txt

TSV files of IMDB reviews used for debugging and grading.

## imdb_practice_result.txt

Desired output for `feature_extraction.py` on `imdb_practice.txt`. 

## util.py

Utility functions for data loading and splitting. 
Includes functions that must be completed for lab and homework.

## lab.py

Creates and shuffles feature vectors.

Usage `python lab.py --path imdb_practice.txt`

## feature_extraction.py

This homework assignment practices creating feature vectors from text. 
Lowercase everything.
The code runs but is missing the parts
that fill the matrix and perform calculations on it. Fill in the missing parts of the code
as directed by the comments. Leave the print statements as they are and replace the variables with calculations,
e.g. calling "array.sum()".

The correct print output for the sample data is saved as imdb_practice_result.txt. Use this to check your work.
For grading, your code will be run on a different data file in the same format and the print-out compared to
the correct results.

