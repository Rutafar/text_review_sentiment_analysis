from src.utils.words import GET_POLARTIY
from src.utils.utils import convert_dict_to_list
from sklearn import svm
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,recall_score

def set_polarity(review):
    word_dictionary = GET_POLARTIY()
    final_sentence = list()
    for sentence in review:
        sentence = sentence.split()
        for i in range(0, len(sentence)):
            if sentence[i] in word_dictionary:
                score = word_dictionary[sentence[i]]
                friend = False
                if i+1<len(sentence) and sentence[i + 1] in word_dictionary :
                    final_sentence.append(score_word(sentence, word_dictionary, i, 1, score))

                if i+2 < len(sentence) and sentence[i + 2] in word_dictionary:
                    final_sentence.append(score_word(sentence, word_dictionary, i, 2, score))

                if not friend:
                    final_sentence.append((sentence[i] , score))

    return final_sentence


def score_word(sentence, word_dictionary, index, pad, score):
    score_tup = 0
    if score < 0 and word_dictionary[sentence[index+pad]] < 0:
        score_tup = score
    elif word_dictionary[sentence[index + pad]]<0:
        score_tup = score * word_dictionary[sentence[index + pad]]
    else:
        score_tup = score * word_dictionary[sentence[index + pad]]

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

