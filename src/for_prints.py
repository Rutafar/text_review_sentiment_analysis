from src.data.import_dataset import import_cleaned_training_set
from src.features.explore import bag_of_words, nouns_adverbs_adjectives
from src.features.normalize import nouns_and_adjectives
from src.visualization.visualize import display_features


dataset = import_cleaned_training_set()
comments_only = [review.reviewText for review in dataset]

print(comments_only[0:5])
nouns_adj = nouns_and_adjectives(comments_only[0:10], 'training')

print(comments_only[2])
print(nouns_adj[2])
n_a_bow_vec, n_a_bow_feat = bag_of_words(nouns_adj)



print(display_features(n_a_bow_feat.todense(), n_a_bow_vec.get_feature_names()))

n_a_a_vec, n_a_a_feat = bag_of_words(nouns_adverbs_adjectives(comments_only[0:10], 'training'))
print(display_features(n_a_a_feat.todense(), n_a_a_vec.get_feature_names()))



print(comments_only[2])
print(nouns_adj[2])