from tqdm import tqdm
import pickle
from os.path import dirname, join,  abspath
_BASIC_PATH = (join(dirname(dirname(dirname(abspath(__file__)))),"data"))



def get_file_path(filename):

    return join(_BASIC_PATH, filename)

def import_cleaned_training_set():
    with open(get_file_path('processed\\lexicon_dataset_small.pkl'), 'rb') as file:
        training = pickle.load(file)

    return training


def import_cleaned_testing_set():
    with open(get_file_path('processed\\testing_sentenced.pkl'), 'rb') as file:
        testing = pickle.load(file)

    return testing


training_imp = import_cleaned_training_set()
testing_imp = import_cleaned_testing_set()
all = list(training_imp) + list(testing_imp)

positive = (open(get_file_path("positive-words.txt"), "r").read())
negative = (open(get_file_path("negative-words.txt"), "r").read())
positive = positive.split()
negative = negative.split()

overall_1 = [review.reviewText for review in all if review.overall == 1]
overall_5 = [review.reviewText for review in all if review.overall==5]

count = dict()
ov_total = overall_1 + overall_5
for review in tqdm(ov_total):
    review = ' '.join(review)
    for word in review.split():

        if word in count:
            count[word] = count[word] + 1
        else:
            if word in positive or word in negative:
                count[word] = 1

import operator
sorted_d = sorted(count.items(), key=operator.itemgetter(1))

new_negative = open(get_file_path("words.txt"), "w")

for word in sorted_d[::-1]:

    if word[1] > 100:
        new_negative.write(word[0] +" : " + str(word[1]) + "\n")
