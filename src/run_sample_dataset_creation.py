from src.utils.utils import get_file_names
from data.make_sample_dataset import read_pickle_files, divide, write_new_pickle


def sampling():
    files = get_file_names()
    training = list()
    testing = list()
    for file in files:
        print("reading file")
        out = read_pickle_files(file)
        print("sampling")
        training_sample, testing_sample = divide(out)
        training = training + training_sample
        testing = testing + testing_sample
        print("to pickle")
    print(training)
    write_new_pickle(training, "training")
    write_new_pickle(testing, "testing")


if __name__ == '__main__':
    sampling()