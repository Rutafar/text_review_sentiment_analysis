import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
from src.utils.utils import get_file_names

from src.utils.utils import get_file_path


def display_features(features, feature_names):
    df = pd.DataFrame(data=features, columns=feature_names)
    return df

def display_scores(scores):
    plt.plot(sorted(scores, key=int, reverse=True)[0:10])
    plt.ylabel("Scores")
    plt.xlabel("Features")
    plt.show()

def plot_confusion_matrix(confusion):
    classes = [1.0, 2.0, 3.0]

    confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    plt.imshow(confusion, cmap=plt.cm.Blues ,interpolation='nearest' )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    fmt = '.2f'
    thresh = confusion.max() / 2.
    for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(j, i, format(confusion[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion[i, j] > thresh else "black")
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.show()


def plot_explained_variance(file_name):
    file = open(get_file_path(file_name), "r")
    total = 0.0
    data = list()
    data_y = list()
    i = 1
    for num in file.read().split():
        total = float(num) + total
        data.append(total)
        data_y.append(i)
        i += 1
    plt.plot(data, data_y)
    plt.ylabel('Number of Components')
    plt.xlabel('Explained Variance')
    plt.title('Bag Of Nouns SVD Components Explained Variance')
    plt.show()


def prepare_dataframe():
    with open(get_file_path("dataframe.pkl"), 'rb') as f:
        df = pickle.load(f)
    new_df = pd.DataFrame(index=['reviews_Automotive', 'reviews_Cell_Phones_and_Accessories', 'reviews_Video_Games', 'reviews_Movies_and_TV'], columns=[0,1,2,3,4,5,6,7,8,9])
    for file in get_file_names():
        for i in range(0,10):
            df_temp = df[df.categories == file]
            m = df_temp[i].mean()
            new_df.at[file, i] = m

    return new_df


def plot_heatmap():
    classes = ['Automotive', 'Cell_Phones', 'Video_Games', 'Movies_and_TV']
    concepts = [0,1,2,3,4,5,6,7,8,9]
    df = prepare_dataframe()
    array_values = df.values
    array_values = array_values.astype(float)

    plt.imshow(array_values, cmap=plt.cm.BuGn, interpolation='nearest')
    plt.ylabel('Categories')
    plt.xlabel('Concepts')

    tick_marks_x = np.arange(len(classes))
    plt.yticks(tick_marks_x, classes)
    plt.xticks(np.arange(len(concepts)), concepts)
    plt.tight_layout()
    plt.show()