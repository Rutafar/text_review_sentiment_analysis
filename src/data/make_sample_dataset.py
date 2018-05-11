import pickle
import pandas as pd
from src.utils.utils import get_file_path


def read_pickle_files(file):
    with open(get_file_path('raw\\' + file+ '.pkl'), 'rb') as lines:
        return pickle.load(lines)


def divide(review_list):
    dataframe = pd.DataFrame(review_list)
    print("overall 1")
    overall_1 = dataframe[dataframe.overall == 1].sample(20000)
    print("overall 2")
    overall_2 = dataframe[dataframe.overall == 2].sample(20000)
    print("overall 3")
    overall_3 = dataframe[dataframe.overall == 3].sample(20000)
    print("overall 4")
    overall_4 = dataframe[dataframe.overall == 4].sample(20000)
    print("overall 5")
    overall_5 = dataframe[dataframe.overall == 5].sample(20000)
    return overall_1.to_dict('records') + overall_2.to_dict('records') + overall_3.to_dict('records') \
           + overall_4.to_dict('records') + overall_5.to_dict('records')




def write_new_pickle(review_list, name):
    with open(get_file_path("interim\\sample_" + name + ".pkl"), "wb") as f:
        pickle.dump(review_list, f)

