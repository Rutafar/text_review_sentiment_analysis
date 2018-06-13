from src.features.sentiment import set_polarity, run_model,extract_unique_words, sum_repeated,  create_sparse_matrix
from src.data.import_dataset import import_cleaned_training_set, import_cleaned_testing_set, read_pickle, import_neutral_negative
from src.data.export_dataset import export_dataset, save_lexicon_results
from src.utils.words import GET_POLARTIY

def main():
    training_imp = import_cleaned_training_set()
    testing_imp = import_cleaned_testing_set()
    training, testing = get_text_from_reviews(training_imp, testing_imp)
    training_overall = extract_overall_from_reviews(training_imp)
    testing_overall = extract_overall_from_reviews(testing_imp)
    print("polarity")
    word_dictionary = extract_word_dictionary()
    #static_dictionary = GET_POLARTIY()
    negative, neutral =import_neutral_negative()
    print(len(word_dictionary))
    polarity_train = [set_polarity(review, word_dictionary, negative, neutral) for review in training]
    polarity_test = [set_polarity(review, word_dictionary, negative, neutral) for review in testing]
    print("removing repeats")
    without_repeat_train = sum_repeated(polarity_train)
    without_repeat_test = sum_repeated(polarity_test)
    print("creating matrix")
    all_words = without_repeat_test + without_repeat_train
    unique_words = extract_unique_words(all_words)
    print(len(unique_words))
    matrix_test = create_sparse_matrix(unique_words, without_repeat_test)
    #save_lexicon_results(matrix_test, 'testing_matrix')
    matrix  = create_sparse_matrix(unique_words, without_repeat_train)
    #save_lexicon_results(matrix, "training_matrix")
    print("svm")
    run_model(matrix, matrix_test, training_overall, testing_overall)


def get_text_from_reviews(training, testing):
    training_text = [review.getreviewText() for review in training]
    testing_text = [review.getreviewText() for review in testing]

    return training_text, testing_text


def extract_overall_from_reviews(dataset):
    overall_only = [review.overall for review in dataset]
    return overall_only


def extract_word_dictionary2():
    lex = read_pickle('', 'sent_lex1000')
    lex_dict=dict()
    for pair in lex:
        lex_dict[pair[0]] = pair[1]
    return lex_dict

def extract_word_dictionary():

    return  read_pickle('', 'filtered_lexicon')


if __name__ == '__main__':
    main()
