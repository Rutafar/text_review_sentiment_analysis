from nltk.corpus import stopwords

from re import IGNORECASE, DOTALL, sub, compile
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from src.utils.contractions import get_contractions


def letters_only(text):
    letters = sub("[^a-zA-Z]", " ", text)
    return letters


def lower_only(text):
    return text.lower().split()


def remove_stopwords(text):
    no_stopwords = [word for word in text if word not in stopwords.words("english")]

    return " ".join(no_stopwords)


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    word_categories = tag_word(text.split())
    lemma_list_of_words =list()
    for i in text.split():

        word = i

        cat = word_categories.get(word)

        if cat == ' ':
            lemma_list_of_words.append(word)
            continue
        else:
            lem = lemmatizer.lemmatize(word, cat)
            lemma_list_of_words.append(lem)

    return lemma_list_of_words


def remove_contractions(normalized_text):
    contraction_mapping = get_contractions()
    contractions_pattern = compile('({})'.format('|'.join(contraction_mapping.keys())), flags=IGNORECASE | DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)
        if not expanded_contraction:
            expanded_contraction =contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    normalized_text = contractions_pattern.sub(expand_match, normalized_text)
    normalized_text = normalized_text.strip()
    return normalized_text



'''
VB - verb
VBP - Verb
NN - noun
JJ - adj
'''


def tag_word(text):
    tags = pos_tag(text)
    word_categories = dict()

    for word in tags:
        t = ' '
        tag = word[1][0]
        if tag == 'N':
            t = wordnet.NOUN
        elif tag == 'V':
            t = wordnet.VERB
        elif tag == 'J':
            t = wordnet.ADJ
        elif tag == 'R':
            t = wordnet.ADV

        word_categories[word[0]] = t
    return word_categories


def clean(text):

    text = letters_only(text)
    text = lower_only(text)
    text = remove_stopwords(text)

    text = remove_contractions(text)
    text = lemmatize(text)
    return ' '.join(text)

