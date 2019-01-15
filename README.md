sta-lda
====

Implementation of the algorithm of the paper [A Reduction for Efficient LDA Topic Reconstruction](https://papers.nips.cc/paper/8012-a-reduction-for-efficient-lda-topic-reconstruction).

Building and installing
----
Download/clone the repository:
```
git clone git@github.com:matteojug/lda-sta.git
cd lda-sta
```
then build it:
```
python setup.py build
```
and install it (optionally with `--user` to install only for the user):
```
python setup.py install
```

To check that everything is working, try running the example (requires `sklearn`):
```
python example.py
```

Doc
----
```
anchor(...)
    Reconstruct the topics from a corpus
    
    Args:
        corpus: The corpus as a list of documents, each being a dict representing the bag of words of the document as integer key/value {wordId:count}.
        K: The number of topics to reconstruct.
        alpha: The dirichlet parameter (usually 1/K).
        c: The anchor cardinality (with c=1 anchors are single words, c=2 pairs, ecc..).
        algId: The algorithm to be used (Default: ALGO_LAZY_CLUSTER).
        algParams: A dict containing the parameters for the algorithm as string key/value (Default: {}).
        vocab: The vocabulary as list of words, useless in non debug mode (Default: None).
    
    Returns:
        A list of topics, each being a probability distribution as a list of real values (one for each word of the vocabulary)
    
    Notes:
    The complexity is exponential in c, using values higher than 1 is not recommended unless with very small vocabulary size; higher values require also to specify a distance function to be used.
    The algorithms available are:
        ALGO_STA_DUMP: dump the STA (Single Topic Allocation) probabilities;
        ALGO_GREEDY: greedy algorithm that picks the best candidates that don't collide with already picked ones;
        ALGO_CLUSTER: perform agglomerive clustering on the candidates;
        ALGO_LAZY_GREEDY/ALGO_LAZY_CLUSTER: lazy implementations where the LDA/STA probabilities are computed on demands (require c=1);
    The recommended ones are ALGO_LAZY_CLUSTER and, if c>1, ALGO_CLUSTER.
    As algParams, the following parameters can be used (the values provided must be casted to string):
        delta, eps: required, refer to the paper;
        p: required if using the greedy algorithms, refer to the paper;
        ngram: (all|sample|prefix, default: all) specify what ngrams are used to compute the LDA n-grams distribution;
        dist: (taud|tau|linf|innerprod, default: taud) the distance function, linf and innerprod refers to the resp. metrics applied to the topics induced by the anchors;
        linkage: (average|single|complete|innerprod, default: average) the linkage policy used by the clustering algorithm;
        candidate_size: force the size of the candidate anchors pool (ignoring the rho cutoff);
        threshold_dist: required if using the greedy algorithm and a distance function other than tau, overwrite the threshold that define two anchors colliding;
        ngram_sample_count: if ngram=sample, the numberd of samples to take from each document;
        random_seed: if ngram=sample, random seed to use when required;
        stadump_path: if algo=STA_DUMP, path where to save the STA distribution;
```
