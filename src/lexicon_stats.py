from src.data.import_dataset import read_pickle
import pandas as pd
from src.utils.utils import get_file_names, get_file_path
from src.data.export_dataset import save_lexicon_results
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.words import GET_POLARTIY

lexicon = read_pickle('', 'lexicon_results')


hu = GET_POLARTIY()

only_values_hu = list()
for key, value in hu.items():
    only_values_hu.append(value)



lexicon_dict = dict()

for pair in lexicon:
    lexicon_dict[pair[0]] = pair[1]

for key, value in hu.items():
    if key in lexicon_dict and lexicon_dict[key] > 0 and value < 0:
        print(key)
        print(value)
        print(lexicon_dict[key])

'''
filtered_dict = dict()
positive = 0
negative = 0
for word in lexicon:

    if word[1]> 1 or word[1] < -1:

        if word[1] > 0 :
            positive = positive + 1
        if word[1] < 0:
            negative = negative + 1

        filtered_dict[word[0]] = word[1]

print(len(filtered_dict))
print(positive)
print(negative)
save_lexicon_results(filtered_dict, 'filtered_lexicon5000')
'''