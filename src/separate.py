from src.data.import_dataset import read_pickle
import pandas as pd
from src.data.export_dataset import export_dataset
_FILE_NAMES = ['reviews_Automotive', 'reviews_Cell_Phones_and_Accessories', 'reviews_Video_Games', 'reviews_Movies_and_TV']

p = read_pickle('interim', 'lexicon_dataset_small')
print(p)
dataframe = pd.DataFrame(p)
ov_total = []
for file in _FILE_NAMES:
    for i in range(1,6):
        overall_1 = dataframe[dataframe.overall == i & dataframe.category == file].sample(75).to_dict('records')
        ov_total = ov_total + overall_1


export_dataset(ov_total, 'lexicon_dataset_smaller')


