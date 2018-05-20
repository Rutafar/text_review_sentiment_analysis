from src.utils.sentiment_words import export_to_dict
from data.import_dataset import read_pickle



data = read_pickle("interim", "sample_reviews_Automotive")
for d in data:
    print(d['reviewText'])
#export_to_dict()