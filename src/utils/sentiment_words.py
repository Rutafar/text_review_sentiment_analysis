from src.utils.utils import get_file_path


def export_to_dict():
    positive = (open(get_file_path("positive-words.txt") , "r").read())
    negative = (open(get_file_path("negative-words.txt"), "r").read())
    pos = dict()
    neg = dict()
    for line in positive.split():
        pos[line] = 1
    for line in negative.split():
        neg[line] = -1
    f = open("words.py", "w")
    f.write("POSITIVE_MAP=" + str(pos) + "\n")
    f.write("NEGATIVE_MAP=" + str(neg))
    f.close()