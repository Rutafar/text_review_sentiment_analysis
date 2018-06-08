from features.normalize import lemmatize

from src.utils.utils import get_file_path


positive = (open(get_file_path("positive-words.txt"), "r").read())
negative = (open(get_file_path("negative-words.txt"), "r").read())
positive = positive.replace("\n", " ")
negative = negative.replace("\n", " ")
pos = lemmatize(positive)
neg = lemmatize(negative)

new_positive = open(get_file_path("positive_lemmatized.txt"), 'w')
pos = sorted(list(set(pos.split())))
for word in pos:
    new_positive.write("%s\n" % word)

new_negative = open(get_file_path("negative_lemmatized.txt"), "w")
neg = sorted(list(set(neg.split())))
for word in neg:
    new_negative.write("%s\n" % word)

