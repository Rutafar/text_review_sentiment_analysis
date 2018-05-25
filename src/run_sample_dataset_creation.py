from src.utils.utils import get_file_names
from src.data.make_sample_dataset import divide, write_new_pickle, divide_estupido
from src.data.import_dataset import  read_pickle


def sampling():
    files = get_file_names()
    training = list()
    testing = list()
    for file in files:
        print("File: " + file)
        out = read_pickle("raw", file)
        print("sampling")
        training_sample, testing_sample = divide(out)
        training = training + training_sample
        testing = testing + testing_sample
        print("to pickle")

    write_new_pickle(training, "training")
    write_new_pickle(testing, "testing")


def sampling_estupido():
    files = get_file_names()
    final = list()
    for file in files:
        print("File: " + file)
        out = read_pickle("raw", file)
        print("sampling")
        stupid = divide_estupido(out)
        final = final + stupid
        print("to pickle")

    write_new_pickle(stupid, "lexicon_dataset")


if __name__ == '__main__':
    sampling_estupido()