from src.data.import_dataset import import_cleaned_training_set, import_cleaned_testing_set
from data.export_dataset import export_nouns_adj_adv
from models.classification_model import model_bag_of_words, model_bigrams
from datetime import datetime
import numpy as np
from features.explore import nouns_adverbs_adjectives
from features.normalize import tag_word


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
    tags_training = tag_word(' '.join(comments_training))
    export_nouns_adj_adv(tags_training, 'tagged_words_training')
    tags_testing = tag_word(' '.join(comments_testing))
    export_nouns_adj_adv(tags_testing, 'tagged_words_testing')


def extract_comments_from_reviews(dataset):
    comments_only = [review.reviewText for review in dataset]
    return comments_only


def extract_categories_from_reviews(dataset):

    categories_only = [review.category for review in dataset]
    return categories_only


def extract_for_wordcloud(dataset):
    movies = [[review.reviewText, review.category] for review in dataset]
    return movies




if __name__ == '__main__':
    main()