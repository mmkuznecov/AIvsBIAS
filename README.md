# AIvsBIAS

This is a project to fit Word2Vec embedding without gender bias.

## Motivation

For a long time, the global cultural background has remained highly gender biased. In the 21st century, all these stereotypes are being adopted by machine learning models trained on statistical data and corpora of texts. Here is how it happens: artificial intelligence works with embeddings - representations of words in the form of vectors, where the distance between the codes of words is related to their semantic similarity. Embeddings are trained on large volumes of text, which can not be called gender-neutral (for example, classic literature). As a result, the algorithms can work incorrectly, matching a lot of not always appropriate terms with the words denoting gender identity.

## Proposed solution

We present a gender-neutral language model whose training is done with the use of a special regularization function that allows models to remain gender-neutral even after training on nonbiased datasets.
