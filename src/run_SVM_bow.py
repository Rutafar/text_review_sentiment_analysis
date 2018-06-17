from src.data.import_dataset import import_cleaned_training_set, import_cleaned_testing_set, import_tagged_words
from src.data.export_dataset import export_dataset
from models.classification_model import model_bag_of_words
from datetime import datetime
import numpy as np
from features.explore import nouns_adverbs_adjectives
from features.normalize import tag_word, nouns_and_adjectives

from visualization.visualize import plot_confusion_matrix


def main():
    start = datetime.now()
    print(start)

    print('Importing training....')
    training = import_cleaned_training_set()

    print('Importing testing....')
    testing = import_cleaned_testing_set()

    print('Taking Out comments')
    comments_training = extract_comments_from_reviews(training)
    comments_testing = extract_comments_from_reviews(testing)

    overall_training = extract_overall_from_reviews(training)
    overall_testing = extract_overall_from_reviews(testing)

    print("------Normal------")

    conf_model = model_bag_of_words(comments_training,comments_testing, overall_training, overall_testing, 5)


    print("------Nouns, Adverbs, Adjectives------")
    n_a_adj_training = nouns_adverbs_adjectives(comments_training, 'training')
    n_a_adj_testing = nouns_adverbs_adjectives(comments_testing, 'testing')
    conf_a_n_adj = model_bag_of_words(n_a_adj_training, n_a_adj_testing, overall_training, overall_testing,2)

    print("------Nouns and Adjectives------")
    nouns_adj_training= nouns_and_adjectives(comments_training, 'training')
    nouns_adj_testing = nouns_and_adjectives(comments_testing, 'testing')
    conf_adj_noun = model_bag_of_words(nouns_adj_training, nouns_adj_testing, overall_training, overall_testing, 4)

    plot_confusion_matrix(conf_model)
    plot_confusion_matrix(conf_a_n_adj)
    plot_confusion_matrix(conf_adj_noun)


def extract_comments_from_reviews(dataset):
    comments_only = [review.reviewText for review in dataset]
    return comments_only


def extract_overall_from_reviews(dataset):
    overall_only = [review.overall for review in dataset]
    return overall_only


def extract_for_wordcloud(dataset):
    movies = [[review.reviewText, review.category] for review in dataset]
    return movies




if __name__ == '__main__':
    main()