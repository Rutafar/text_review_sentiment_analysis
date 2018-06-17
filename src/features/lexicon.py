import sys
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
from numpy import dot, sqrt, array

def cooccurrence_matrix(corpus):
    """
    Create the co-occurrence matrix.

    Input
    corpus (tuple of tuples) -- tokenized texts

    Output
    d -- a two-dimensional defaultdict mapping word pairs to counts
    """
    d = defaultdict(lambda : defaultdict(int))
    for text in corpus:
        for i in range(len(text)-1):
            for j in range(i+1, len(text)):

                w1, w2 = sorted([text[i], text[j]])
                d[w1][w2] += 1
    return d

def get_sorted_vocab(d):
    """
    Sort the entire vocabulary (keys and keys of their value
    dictionaries).

    Input
    d -- dictionary mapping word-pairs to counts, created by
         cooccurrence_matrix(). We need only the keys for this step.

    Output
    vocab -- sorted list of strings
    """
    vocab = set([])
    for w1, val_dict in d.items():
        vocab.add(w1)
        for w2 in val_dict.keys():
            vocab.add(w2)
    vocab = sorted(list(vocab))
    return vocab


def cosine_similarity_matrix(vocab, d):
    """
    Create the cosine similarity matrix.

    Input
    vocab -- a list of words derived from the keys of d
    d -- a two-dimensional defaultdict mapping word pairs to counts,
    as created by cooccurrence_matrix()

    Output
    cm -- a two-dimensional defaultdict mapping word pairs to their
    cosine similarity according to d
    """
    cm = defaultdict(dict)
    vectors = get_vectors(d, vocab)
    for w1 in tqdm( vocab):
        for w2 in vocab:
            cm[w1][w2] = cosim(vectors[w1], vectors[w2])
    return cm

def get_vectors(d, vocab):
    """
    Interate through the vocabulary, creating the vector for each word
    in it.

    Input
    d -- dictionary mapping word-pairs to counts, created by
         cooccurrence_matrix()
    vocab -- sorted vocabulary created by get_sorted_vocab()

    Output
    vecs -- dictionary mapping words to their vectors.
    """
    vecs = {}
    for w1 in vocab:
        v = []
        for w2 in vocab:
            wA, wB = sorted([w1, w2])
            v.append(d[wA][wB])
        vecs[w1] = array(v)
    return vecs


def cosim(v1, v2):
    """Cosine similarity between the two vectors v1 and v2."""
    num = dot(v1, v2)
    den = sqrt(dot(v1, v1)) * sqrt(dot(v2, v2))
    if den:
        return num/den
    else:
        return 0.0

def graph_propagation(cm, vocab, positive, negative, iterations):
    """
    The propagation algorithm employing the cosine values.

    Input
    cm -- cosine similarity matrix (2-d dictionary) created by cosine_similarity_matrix()
    vocab -- vocabulary for cm
    positive -- list of strings
    negative -- list of strings
    iterations -- the number of iterations to perform

    Output:
    pol -- a dictionary form vocab to floats
    """
    pol = {}
    # Initialize a.
    a = defaultdict(lambda : defaultdict(int))
    for w1, val_dict in cm.items():
        for w2 in val_dict.keys():
            if w1 == w2:
                a[w1][w2] = 1.0
    # Propagation.
    print('propagate')
    pol_positive, a = propagate(positive, cm, vocab, a, iterations)
    pol_negative, a = propagate(negative, cm, vocab, a, iterations)
    print('beta')
    beta = sum(pol_positive.values()) / sum(pol_negative.values())
    for w in tqdm(vocab):
        pol[w] = pol_positive[w] - (beta * pol_negative[w])
    return pol


def propagate(seedset, cm, vocab, a, iterations):
    """
    Propagates the initial seedset, with the cosine measures
    determining strength.

    Input
    seedset -- list of strings.
    cm -- cosine similarity matrix
    vocab -- the sorted vocabulary
    a -- the new value matrix
    iterations -- the number of iteration to perform

    Output
    pol -- dictionary mapping words to un-corrected polarity scores
    a -- the updated matrix
    """
    for w_i in tqdm(seedset):
        f = {}
        f[w_i] = True
        for t in range(iterations):
            for w_k in cm.keys():
                if w_k in f:
                    for w_j, val in cm[w_k].items():
                        # New value is max{ old-value, cos(k, j) } --- so strength
                        # can come from somewhere other th
                        a[w_i][w_j] = max([a[w_i][w_j], a[w_i][w_k] * cm[w_k][w_j]])
                        f[w_j] = True
    # Score tally.
    pol = {}
    for w in vocab:
        pol[w] = sum(a[w_i][w] for a_i in seedset)
    return [pol, a]


def format_matrix(vocab, m):
    """
    For display purposes: builds an aligned and neatly rounded version
    of the two-dimensional dictionary m, assuming ordered values
    vocab. Returns string s.
    """
    s = ""
    sep = ""
    col_width = 15
    s += " ".rjust(col_width) + sep.join(map((lambda x : x.rjust(col_width)), vocab)) + "\n"
    for w1 in vocab:
        row = [w1]
        row += [round(m[w1][w2], 2) for w2 in vocab]
        s += sep.join(map((lambda x : str(x).rjust(col_width)), row)) + "\n"
    return s