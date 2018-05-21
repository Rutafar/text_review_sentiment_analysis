from features.lexicon import cosim,cosine_similarity_matrix,format_matrix,get_sorted_vocab,get_vectors,graph_propagation,propagate
from operator import itemgetter
import dill as pickle
from collections import defaultdict
from src.data.import_dataset import import_cleaned_training_set, import_cleaned_testing_set
from src.data.export_dataset import save_lexicon_results
from datetime import datetime


from src.utils.utils import get_file_path

def main():
    start = datetime.now()
    print(start)

    print('Importing training....')
    training = import_cleaned_training_set()
    testing = import_cleaned_testing_set()

    print('Taking Out comments')
    training = extract_comments_from_reviews(training)
    testing = extract_overall_from_reviews(testing)
    print("-----Creating training lexicon-----")
    create_lexicon(training, 'training')
    print("-----Creating testing lexicon-----")
    create_lexicon(testing, 'testing')

    print(datetime.now() - start)




def create_lexicon(corpus, name):
    print("Cooccurrence matrix")
    d = cooccurrence_matrix(corpus)
    print("Sorting vocab")
    vocab = get_sorted_vocab(d)
    with open(get_file_path(name + '.pkl'), 'wb') as f:
        pickle.dump([d,vocab], f, pickle.HIGHEST_PROTOCOL)

    print("Cosine matrix")
    cm = cosine_similarity_matrix(vocab, d)
    save_lexicon_results({"cooccurrence": d, "vocabulary": vocab, "matrix": cm}, name)

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
        text = text.split()
        for i in range(len(text)-1):
            for j in range(i+1, len(text)):

                w1, w2 = sorted([text[i], text[j]])
                d[w1][w2] += 1
    return d


def print_matrixes(vocab, d, cm):
    print("Co-occurence matrix:\n")
    print(format_matrix(vocab, d))
    print("Cosine similarity matrix:\n")
    print(format_matrix(vocab, cm))

    
def extract_comments_from_reviews(dataset):
    comments_only = [review.reviewText for review in dataset]
    return comments_only


def extract_overall_from_reviews(dataset):
    overall_only = [review.overall for review in dataset]
    return overall_only



if __name__ == '__main__':
    main()