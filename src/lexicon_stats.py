from src.data.import_dataset import read_pickle
import pandas as
from src.utils.utils import get_file_names, get_file_path


lexicon = read_pickle('', 'lexicon_results')

print(len(lexicon))

positives = 0
negatives = 0
max_positive = 0
max_negative = 0
for pair in lexicon:
    if pair[1]<0:
        if pair[1]< max_negative:
            max_negative = pair[1]
        negatives = negatives + 1
    else:
        if pair[1]> max_positive:
            max_positive= pair[1]
        positives = positives +1

print(positives, negatives, max_negative, max_positive)
words = []
for pair in lexicon:
    words.append(pair[0])

new_negative = open(get_file_path("lexicon.txt"), "w")

for word in sorted(words):
        new_negative.write(word + "\n")
