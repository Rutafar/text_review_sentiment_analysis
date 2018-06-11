from src.utils.words import GET_POLARTIY
from src.utils.utils import convert_dict_to_list
from sklearn import svm
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,recall_score
from tqdm import tqdm
from scipy.sparse import coo_matrix
import  numpy as np

def set_polarity(review, word_dictionary):

    final_sentence = list()
    for sentence in review:
        sentence = sentence.split()
        for i in range(0, len(sentence)):
            if sentence[i] in word_dictionary:
                score = word_dictionary[sentence[i]]

                friend = False
                if i+1<len(sentence) and sentence[i + 1] in word_dictionary :
                    final_sentence.append(score_word(sentence, word_dictionary, i, 1, score))
                    friend=True

                if i+2 < len(sentence) and sentence[i + 2] in word_dictionary:
                    final_sentence.append(score_word(sentence, word_dictionary, i, 2, score))
                    friend=True
                if not friend:
                    final_sentence.append((sentence[i] , score))

    return final_sentence


def score_word(sentence, word_dictionary, index, pad, score):
    score_tup = 0
    if  word_dictionary[sentence[index + pad]]<0:
        score_tup = score * -1
    else:
        score_tup = score
    return sentence[index] + "_" + sentence[index + pad], score_tup


def run_model(training,testing, training_categories, testing_categories):
    clf = svm.SVC(kernel='linear', C=1.0)
    print('Fitting')
    clf.fit(training, training_categories)
    print(datetime.now())
    print('Predicting')
    predicted = clf.predict(testing)
    print(datetime.now() )
    print("Accuracy: " + str(accuracy_score(testing_categories, predicted)))


def sum_repeated(dataset):
    final = dict()
    list_without_repeat = list()
    for review in dataset:
        final.clear()
        for pair in review:
            if pair[0] in final:
                 final[pair[0]] = final[pair[0]] + pair[1]
            else:
                final[pair[0]] = pair[1]

        list_without_repeat.append(convert_dict_to_list(final))

    return list_without_repeat


def extract_unique_words(list_of_tuples):
    unique_words = set()
    for review in list_of_tuples:
        for pair in review:
            unique_words.add(pair[0])
    return sorted(list(unique_words))


def create_matrix(all_words, review_list):
    word_list = extract_unique_words(all_words)
    for_svm = list()
    size = len(word_list)

    for review in tqdm(review_list):
        one_review = [0] * size
        for pair in review:
            index = word_list.index(pair[0])
            one_review[index] = pair[1]

        for_svm.append(one_review)
        
    return for_svm


def create_sparse_matrix(unique_words, dataset):
    columns = []
    rows = []
    data = []
    dataset_size = len(dataset)
    unique_words_size = len(unique_words)
    print(len(unique_words))
    for i in tqdm(range(0, dataset_size)):
        for pair in dataset[i]:
            index = unique_words.index(pair[0])
            columns.append(index)
            rows.append(i)
            data.append(pair[1])
    data = np.asarray(data)
    rows = np.asarray(rows)
    columns = np.asarray(columns)
    matrix = coo_matrix((data, (rows, columns)),shape=(dataset_size, unique_words_size))

    return matrix