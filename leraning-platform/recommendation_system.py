import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
data = pd.read_csv('interactions.csv')

# Preprocess data
content = data['content']
user_interactions = data['user_id']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(content)

# Implement content-based algorithm
cosine_sim = linear_kernel(X, X)

# Get user recommendation
user_id = 1
user_interactions = user_interactions[user_id]
user_interactions = user_interactions.tolist()
user_interactions = set(user_interactions)
user_interactions = list(user_interactions)

recommendations = []
for i in range(len(cosine_sim[user_id])):
    if i not in user_interactions:
        recommendations.append((i, cosine_sim[user_id][i]))

recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

# Format and return recommendations
recommendations = recommendations[:10]
formatted_recommendations = []
for recommendation in recommendations:
    formatted_recommendations.append((vectorizer.get_feature_names()[recommendation[0]], recommendation[1]))

# Save the model
import pickle

filename = 'model.sav'
pickle.dump(vectorizer, open(filename, 'wb'))
