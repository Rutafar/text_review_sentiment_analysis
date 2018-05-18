from features.lexicon import cooccurrence_matrix,cosim,cosine_similarity_matrix,format_matrix,get_sorted_vocab,get_vectors,graph_propagation,propagate
from operator import itemgetter
from src.data.import_dataset import import_cleaned_training_set, import_cleaned_testing_set
from datetime import datetime

def main():
    start = datetime.now()
    print(start)

    print('Importing training....')
    training = import_cleaned_training_set()

    print('Taking Out comments')
    corpus = extract_comments_from_reviews(training)

    print("Cooccurrence matrix")
    d = cooccurrence_matrix(corpus)
    print("Sorting vocab")
    vocab = get_sorted_vocab(d)
    print("Cosine matrix")
    cm = cosine_similarity_matrix(vocab, d)

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