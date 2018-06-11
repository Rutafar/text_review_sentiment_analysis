from src.features.sentiment import set_polarity, run_model,extract_unique_words, sum_repeated, create_matrix, create_sparse_matrix
from src.data.import_dataset import import_cleaned_training_set, import_cleaned_testing_set, read_pickle
from src.data.export_dataset import export_dataset, save_lexicon_results

def main():
    training_imp = import_cleaned_training_set()
    testing_imp = import_cleaned_testing_set()
    training, testing = get_text_from_reviews(training_imp, testing_imp)
    training_overall = extract_overall_from_reviews(training_imp)
    testing_overall = extract_overall_from_reviews(testing_imp)
    print("polarity")
    word_dictionary = extract_word_dictionary()
    polarity_train = [set_polarity(review, word_dictionary) for review in training]
    polarity_test = [set_polarity(review, word_dictionary) for review in testing]
    #print(polarity_train)
    print("removing repeats")
    without_repeat_train = sum_repeated(polarity_train)
    without_repeat_test = sum_repeated(polarity_test)

    print("creating matrix")
    all_words = without_repeat_test + without_repeat_train

    #export_dataset([without_repeat_train, without_repeat_test, all_words], "before_matrix")

    unique_words = extract_unique_words(all_words)
    matrix_test = create_sparse_matrix(unique_words, without_repeat_test)
    save_lexicon_results(matrix_test, 'testing_matrix')
    matrix  = create_sparse_matrix(unique_words, without_repeat_train)
    save_lexicon_results(matrix, "training_matrix")
    print("svm")
    #run_model(training_matrix, testing_matrix, training_overall, testing_overall)


def get_text_from_reviews(training, testing):
    training_text = list()
    testing_text = list()
    for review in training:
        training_text.append(review.getreviewText())
    for review in testing:
        testing_text.append(review.getreviewText())
    return training_text, testing_text


def transform_for_svm(training):
    training_new=list()
    for review in training:
        inter = [pair[1] for pair in review]
        training_new.append(inter)
    return training_new


def extract_overall_from_reviews(dataset):
    overall_only = [review.overall for review in dataset]
    return overall_only

def extract_word_dictionary():
    return  read_pickle('', 'filtered_lexicon')


if __name__ == '__main__':
    main()
