from features.sentiment import set_polarity, run_model, sum_repeated
from data.import_dataset import import_cleaned_training_set, import_cleaned_testing_set
from data.export_dataset import export_to_txt

def main():
    training_imp = import_cleaned_training_set()
    testing_imp = import_cleaned_testing_set()
    training, testing = get_text_from_reviews(training_imp, testing_imp)
    training_overall = extract_overall_from_reviews(training_imp)
    testing_overall = extract_overall_from_reviews(testing_imp)
    print("polarity")
    polarity_train = [set_polarity(review) for review in training[0:100]]
    polarity_test = [set_polarity(review) for review in testing[0:100]]
    print("svm")
    without_repeat_train = sum_repeated(polarity_train)
    without_repeat_test = sum_repeated(polarity_test)
    #print(without_repeat_train)
    transform_train = transform_for_svm(without_repeat_train)
    transform_test = transform_for_svm(without_repeat_test)
    run_model(transform_train, transform_test, training_overall[0:100], testing_overall[0:100])
    #print(polarity)


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

if __name__ == '__main__':
    main()