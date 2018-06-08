from features.lexicon import cooccurrence_matrix, cosim,cosine_similarity_matrix,format_matrix,get_sorted_vocab,get_vectors,graph_propagation,propagate
from operator import itemgetter
import dill as pickle
from collections import defaultdict
from src.data.import_dataset import import_cleaned_training_set, import_cleaned_testing_set
from src.data.export_dataset import save_lexicon_results
from datetime import datetime
import pandas as pd


from src.utils.utils import get_file_path

def main():
    start = datetime.now()
    print(start)
    print('Importing training....')
    training = import_cleaned_training_set()
    print('Taking Out comments')
    training = extract_comments_from_reviews(training)
    training = sentence_to_word(training)
    print('Creating Tuples')
    training_tuples = tuple(tuple(x) for x in training[0:500])

    print("-----Creating training lexicon-----")
    create_lexicon(training_tuples, 'lexicon_dict')
    print(datetime.now() - start)





def create_lexicon(corpus, name):
    positive = (open(get_file_path("positive_lemmatized.txt"), "r").read())
    negative = (open(get_file_path("negative_lemmatized.txt"), "r").read())
    print("Cooccurrence matrix")
    d = cooccurrence_matrix(corpus)
    print("Sorting vocab")
    vocab = get_sorted_vocab(d)
    print("Cosine matrix")
    cm = cosine_similarity_matrix(vocab, d)
    print(datetime.now())
    print("Propagation ")
    prop = graph_propagation(cm, vocab, positive.split(), negative.split(), 2)
    print(datetime.now())
    final = list()
    for key, val in sorted(prop.items(), key=itemgetter(1), reverse=True):
        final.append((key, val))
    print(datetime.now())
    print("Saving")
    save_lexicon_results(final, "lexicon_results")
    print(datetime.now())


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

def sentence_to_word(dataset):
    word_separated = list()
    for comment in dataset:
        tokens = [sentence.split() for sentence in comment]

        word_separated += tokens
    return word_separated

if __name__ == '__main__':
    main()