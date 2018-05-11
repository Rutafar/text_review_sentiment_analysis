from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,recall_score
from features.explore import bag_of_words, tf_idf
from sklearn import svm
from datetime import datetime
import numpy as np
from visualization.visualize import plot_confusion_matrix


def lsa(matrix, comppnents):

    ls = TruncatedSVD(n_components=comppnents)
    print('fitting')
    fit = ls.fit_transform(matrix)
    print(str(ls.explained_variance_ratio_.sum()))
    return ls, fit


def select_features(nr_features, training_set, training_labels):
    selector = SelectKBest(k=nr_features)
    transformed_set = selector.fit_transform(training_set, training_labels)
    return selector, transformed_set


def model_bag_of_words(training, testing, training_cat, testing_cat):
    bow_vectorizer_training, bow_features_training = bag_of_words(training)
    bow_vectorizer_testing, bow_features_testing = bag_of_words(testing)

    tf, idf_train = tf_idf(bow_features_training)
    tf, idf_test = tf_idf(bow_features_testing)

    ls, reduced_training = lsa(idf_train, 100)
    ls_test, reduced_testing = lsa(idf_test, 100)

    selector, training_features = select_features(3, reduced_training, training_cat)
    selector_test, testing_features = select_features(3, reduced_testing, testing_cat)

    train_model(training_features, training_cat, testing_features, testing_cat)


def model_bigrams(comments_training, comments_testing, categories_training, categories_testing):
    bow_vec_train_big, bow_feat_train_big = bag_of_words(comments_training, 2)
    bow_vec_test_big, bow_feat_test_big = bag_of_words(comments_testing, 2)

    print('TF-IDF')
    tf_big, idf_big = tf_idf(bow_feat_train_big)
    tf_big_t, idf_big_t = tf_idf(bow_feat_test_big)

    print('LSA')
    ls_big_train, ls_reduced_train_big = lsa(idf_big, 50)
    ls_big_test, ls_reduced_test_big = lsa(idf_big_t, 50)

    print('Best Features')
    selector, s_big = select_features(5, ls_reduced_train_big, categories_training)
    selector_test, s_big_t = select_features(5, ls_reduced_test_big, categories_testing)

    print('MODEL BIGRAMS')
    train_model(s_big, categories_training, s_big_t, categories_testing)


def train_model(training, training_categories, test, test_categories):
    start = datetime.now()
    print(start)
    clf = svm.SVC(kernel='linear', C=1.0)
    print('Fitting')
    clf.fit(training, training_categories)
    print(datetime.now()-start)
    print('Predicting')
    predicted = clf.predict(test)
    print(datetime.now()-start)
    print("Accuracy: " + str(accuracy_score(test_categories, predicted)))
    confusion = confusion_matrix(test_categories, predicted)
    plot_confusion_matrix(confusion)
    precision = precision_score(test_categories, predicted, average='macro')
    recall = recall_score(test_categories, predicted, average='macro')
    f_cenas = np.round((2*precision*recall)/(precision+recall),2)
    print("F Cenas " + str(f_cenas))
    print("Recall " + str(recall))
    print("Precision " + str(precision))