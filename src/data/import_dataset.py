import pickle
from src.utils.utils import get_file_names, get_file_path
from data.export_dataset import export_training_testing
from numpy.random import choice
from tqdm import tqdm
import pandas as pd


def import_and_divide():
    files = get_file_names()
    training = list()
    testing = list()
    for file in files:
        with open(get_file_path('interim\\sample_' + file + '.pkl'), 'rb') as f:
            lines = pickle.load(f)
            t = choice(lines, size=70000, replace=False)
            for l in tqdm(t):
                lines.remove(l)
                l['category'] = file
                training.append(l)
            for l in lines:
                l['category'] = file
                testing.append(l)


    export_training_testing(training, testing)

def import_with_overall():
    files = get_file_names()
    training = list()
    testing = list()
    for file in files:
        with open(get_file_path('interim\\sample_' + file + '.pkl'), 'rb') as f:
            lines = pickle.load(f)
            df = pd.DataFrame(lines)
            ov_1 = df[df.overall == 1 ].sample(14000)
            df.drop(ov_1.index)
            ov_2 = df[df.overall == 2].sample(14000)
            df.drop(ov_2.index)
            ov_3 = df[df.overall == 3].sample(14000)
            df.drop(ov_3.index)
            ov_4 = df[df.overall == 4].sample(14000)
            df.drop(ov_4.index)
            ov_5 = df[df.overall == 5].sample(14000)


def import_set():
    with open(get_file_path('interim\\training.pkl'), 'rb') as f:
        training = pickle.load(f)

    with open(get_file_path('interim\\testing.pkl'), 'rb') as f:
        testing = pickle.load(f)

    return training, testing


def import_cleaned_training_set():
    with open(get_file_path('processed\\training.pkl'), 'rb') as file:
        training = pickle.load(file)

    return training

def import_cleaned_testing_set():
    with open(get_file_path('processed\\testing.pkl'), 'rb') as file:
        testing = pickle.load(file)

    return testing