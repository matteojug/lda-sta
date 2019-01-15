"""
From scikit-learn's "Topic extraction with
Non-negative Matrix Factorization and Latent Dirichlet Allocation"
example.
"""

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

import lda_sta

vocab_size = 1000
n_top_words = 10

def print_top_words(topics, vocab):
    for topic_idx, topic in enumerate(topics):
        message = "Topic #%d: " % topic_idx
        message += " ".join(["%s" % vocab[i]
        # message += " ".join(["%s [%.3f]" % (vocab[i], topic[i])
                             for i in sorted(range(len(topic)), key=lambda x: -topic[x])[:n_top_words]])
        print(message)
    print()

def csr_to_bow(M):
    ret = []
    for row in M:
        bow = {}
        for col, val in zip(row.indices, row.data):
            bow[col] = val
        ret.append(bow)
    return ret

print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data
print("Got %d docs" % len(data_samples))
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.8, min_df=5, max_features=vocab_size, stop_words='english')
t0 = time()
corpus_csr = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
vocab = tf_vectorizer.get_feature_names()
print()

corpus = csr_to_bow(corpus_csr)

print("Running anchor (c=1)")
algParams = {"delta": "0.01", "eps": "0.1"}
t0 = time()
topics = lda_sta.anchor(corpus, 10, 0.1, 1, lda_sta.ALGO_LAZY_CLUSTER, algParams)
print("done in %0.3fs." % (time() - t0))
print_top_words(topics, vocab)
