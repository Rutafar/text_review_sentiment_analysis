from src.data.import_dataset import read_pickle
import pandas as pd
from src.utils.utils import get_file_names, get_file_path
from src.data.export_dataset import save_lexicon_results
import matplotlib.pyplot as plt
import seaborn as sns


lexicon = read_pickle('', 'lexicon_results')
before_matrix = read_pickle('', 'before_matrix')

print(len(before_matrix[0]))

only_values = list()
unique_words = set()
for pair in lexicon:
    only_values.append(pair[1])


only_values = sorted(only_values)
print(only_values)
plt.plot(only_values)
plt.show()
only_values = [i for i in only_values if i >2 or i < -2]
sns.distplot(only_values)


filtered = [number for number in only_values if number >4 or number < -4]
print(len(filtered))
save_lexicon_results(filtered, "filtered_lexicon")

filtered_dict = dict()
positive = 0
negative = 0
for word in lexicon:

    if word[1]> 4 or word[1] < -4:
        if word[1] > 0 :
            positive = positive + 1
        if word[1] < 0:
            negative = negative + 1
        filtered_dict[word[0]] = word[1]

print(len(filtered_dict))
print(positive)
print(negative)
save_lexicon_results(filtered_dict, 'filtered_lexicon')
