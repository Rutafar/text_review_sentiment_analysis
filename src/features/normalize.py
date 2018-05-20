from nltk.corpus import stopwords
from tqdm import tqdm
from re import IGNORECASE, DOTALL, sub, compile
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, sent_tokenize
from nltk.corpus import wordnet
from src.utils.dictionaries import get_contractions
from src.data.import_dataset import import_tagged_words, get_stopwords


def letters_only(text):
    letters = sub("[^a-zA-Z]", " ", text)
    return letters


def lower_only(text):
    return text.lower().split()


def remove_stopwords(text):
    no_stopwords = [word for word in text if word not in stopwords.words("english")]

    return " ".join(no_stopwords)


def remove_custom_stopwords(text):
    stop = get_stopwords()
    no_stopwords = [word for word in text if word not in stop]
    return " ".join(no_stopwords)

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    word_categories = tag_word(text)
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
    tags = pos_tag(text.split())

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


def remove_whitespaces(text):
    normalized_text = sub(compile('{}'.format(r'\s+')), ' ', text)
    return normalized_text.strip()

def sentence_tokenize(text):
    return sent_tokenize(text)


def nouns_and_adjectives(comments, dataset_type):
    tags = import_tagged_words(dataset_type)
    final_list = list()
    for sentence in tqdm(comments):
        tokenized = sentence.split()
        u = list()
        size = len(tokenized)
        for i in range(0, len(tokenized)):
            word = tokenized[i]
            if tags[word] == wordnet.ADJ:
                u.append(word)
                if i+1 < size and tags[tokenized[i+1]] == wordnet.NOUN:
                    u.append(word + '_' +tokenized[i+1])
                if i+2 < size and tags[tokenized[i+2]] == wordnet.NOUN:
                    u.append(word + '_' +tokenized[i + 2])
        final_list.append(' '.join(u))

    return final_list


def clean(text):

    text = remove_contractions(text)
    text = lower_only(text)
    text = letters_only(' '.join(text))
    text = remove_whitespaces(text)
    text = lemmatize(text)
    text = remove_custom_stopwords(text)
    return text



