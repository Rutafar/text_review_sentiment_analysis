from nltk import bigrams
from nltk.corpus import wordnet
from src.data.import_dataset import import_tagged_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from features.normalize import tag_word
from tqdm import tqdm


def bag_of_words(data, grams=1):
    vectorizer = CountVectorizer(ngram_range=(grams,grams))
    features = vectorizer.fit_transform(data)
    return vectorizer, features


def tf_idf(bow):
    tfidf = TfidfTransformer(smooth_idf=True,use_idf=True)
    idfs = tfidf.fit_transform(bow)

    return tfidf, idfs


def bigrm(text):
    bigrm = bigrams(text.split())
    print (*map(' '.join, bigrm), sep=', ')


def nouns_adverbs_adjectives(data, dataset):
    wanted_words=list()
    tagged_words = import_tagged_words(dataset)

    for text in tqdm(data):
        wanted = [w for w in text.split() if tagged_words[w] == wordnet.NOUN or tagged_words[w]==wordnet.ADJ
                  or tagged_words[w] == wordnet.ADV]
        wanted_words.append(wanted)

    return wanted_words

def generate_concepts(components,feature_names):
    for i, comp in enumerate(components):
        termsInComp = zip(feature_names, comp)
        sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:10]
        print("Concept %d:" % i)
        for term in sortedTerms:
            print(term[0])
        print(" ")