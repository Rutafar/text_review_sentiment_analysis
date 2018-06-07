import pickle
import pandas as pd
from src.utils.utils import get_file_path


def divide(review_list):
    dataframe = pd.DataFrame(review_list)
    training = list()
    testing = list()
    for i in range(1,6):
        print("overall " + str(i))
        tr, te = sample_overall(dataframe,i)
        training = training + tr
        testing = testing + te

    return training, testing

def divide_estupido(review_list):
    dataframe = pd.DataFrame(review_list)
    final = list()
    for i in range(1,6):
        print("overall " + str(i))
        te = sample_overall_stupid(dataframe, i)
        final = final + te

    return final

def sample_overall(dataframe, ov):
    overall_sample = dataframe[dataframe.overall == ov].sample(20000)
    training = overall_sample.head(14000).to_dict('records')
    testing = overall_sample.tail(6000).to_dict('records')
    return training, testing


def sample_overall_stupid(dataframe, ov):
    overall_s = dataframe[dataframe.overall == ov].sample(140).to_dict('records')
    return overall_s

def write_new_pickle(review_list, name):
    with open(get_file_path("interim\\" + name + ".pkl"), "wb") as f:
        pickle.dump(review_list, f)

