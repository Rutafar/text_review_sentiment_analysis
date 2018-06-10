import pickle
from src.utils.utils import get_file_path
import pickle


def export_dataset(set_to_save, name):
    with open(get_file_path("processed\\"+name + ".pkl"), "wb") as file:
        pickle.dump(set_to_save, file, pickle.HIGHEST_PROTOCOL)


def export_comments(set_to_save):
    with open(get_file_path("processed\\comments.pkl"), "wb") as file:
        pickle.dump(set_to_save, file, pickle.HIGHEST_PROTOCOL)


def export_training_testing(training, testing):
    with open(get_file_path("interim\\training.pkl"), "wb") as file:
        pickle.dump(training, file, pickle.HIGHEST_PROTOCOL)

    with open(get_file_path("interim\\testing.pkl"), "wb") as file:
        pickle.dump(testing, file, pickle.HIGHEST_PROTOCOL)


def export_to_txt(scores, name):
    with open(get_file_path(name + ".txt"), "w") as file:
        for t in scores:
            file.write(' '.join(str(s) for s in t) + '\n')


def export_nouns_adj_adv(data, filename):
    with open(get_file_path('processed\\' +filename + '.pkl'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def save_lexicon_results(results, name):
    with open(get_file_path(name+'.pkl'), 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
