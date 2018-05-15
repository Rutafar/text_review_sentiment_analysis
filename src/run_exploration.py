from src.data.import_dataset import import_cleaned_training_set, import_cleaned_testing_set
from src.data.export_dataset import export_dataset
from models.classification_model import model_bag_of_words, model_bigrams
from tqdm import tqdm
from datetime import datetime
import numpy as np
from nltk import pos_tag
from features.explore import nouns_adverbs_adjectives
from features.normalize import tag_word, nouns_and_adjectives


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
    '''
      print('Tagging training')
      export_dataset(tag_word(' '.join(comments_training)), 'tagged_words_training')
      print('Tagging testing')
      export_dataset(tag_word(' '.join(comments_testing)), 'tagged_words_testing')
      '''

    overall_training = extract_overall_from_reviews(training)
    overall_testing = extract_overall_from_reviews(testing)
    n_a_adj_training = nouns_adverbs_adjectives(comments_training, 'training')
    n_a_adj_testing = nouns_adverbs_adjectives(comments_testing, 'testing')



def extract_comments_from_reviews(dataset):
    comments_only = [review.reviewText for review in dataset]
    return comments_only


def extract_for_wordcloud(dataset):
    movies = [[review.reviewText, review.category] for review in dataset]
    return movies




if __name__ == '__main__':
    main()