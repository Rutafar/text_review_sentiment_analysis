from src.utils.utils import get_file_path


def export_to_dict():
    positive = (open(get_file_path("positive-words.txt") , "r").read())
    negative = (open(get_file_path("negative-words.txt"), "r").read())
    words = dict()
    for line in positive.split():
        words[line] = 1
    for line in negative.split():
        words[line] = -1
    f = open("words.py", "w")
    f.write("WORD_SENTIMENT=" + str(words))
    f.close()
