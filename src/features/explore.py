from nltk import bigrams
from nltk.corpus import wordnet
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


def nouns_adverbs_adjectives(data):
    text = ' '.join(data)
    tags = tag_word(text.split())
    nouns = [word for word in text.split() if tags[word] == wordnet.NOUN or tags[word]== wordnet.ADV or tags[word]== wordnet.ADJ]

    return ' '.join(nouns)


def generate_concepts(components,feature_names):
    for i, comp in enumerate(components):
        termsInComp = zip(feature_names, comp)
        sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:10]
        print("Concept %d:" % i)
        for term in sortedTerms:
            print(term[0])
        print(" ")