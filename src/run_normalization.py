from features.normalize import clean
from src.data.import_dataset import import_set, import_lexicon_set
from tqdm import tqdm
from src.data.export_dataset import export_dataset
from review.Review import create_review_from_sample
import dill


def main():
    training = import_lexicon_set()
    cleaned_review_training_set = clean_sets(training)
    export_dataset(cleaned_review_training_set, 'lexicon_dataset_small')

    #cleaned_review_testing_set = clean_sets(testing)
    #export_dataset(cleaned_review_testing_set, 'testing_sentenced')


def clean_sets(set_to_clean):
    cleaned_set = set()
    for review in tqdm(set_to_clean):
        text = clean(review['reviewText'])

        review['reviewText'] = text

        r = create_review_from_sample(review)
        cleaned_set.add(r)

    return cleaned_set


if __name__ == '__main__':
    main()


