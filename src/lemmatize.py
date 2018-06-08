from features.normalize import lemmatize

from src.utils.utils import get_file_path


positive = (open(get_file_path("positive-words.txt"), "r").read())
negative = (open(get_file_path("negative-words.txt"), "r").read())
positive = positive.replace("\n", " ")
negative = negative.replace("\n", " ")
pos = lemmatize(positive)
print(len(list(set(pos.split()))))
print(len(positive.split()))
neg = lemmatize(negative)
new_positive = open(get_file_path("positive_lemmatized.txt"), 'w')
for word in positive:
    new_positive.write("%s\n" % word)

new_negative = open(get_file_path("negative_lemmatized.txt"), "w")
for word in negative:
    new_negative.write("%s\n" % word)

print(len(list(set(neg.split()))))
print(len(negative.split()))
