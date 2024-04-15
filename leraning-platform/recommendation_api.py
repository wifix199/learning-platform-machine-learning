from flask import Flask, jsonify
from model_loader import vectorizer
from recommendation_system import get_recommendations

app = Flask(__name__)

@app.route('/recommendations/<user_id>')
def get_recommendations_route(user_id):
    data = pd.read_csv('interactions.csv')
    user_interactions = data['user_id']
    user_interactions = user_interactions.tolist()
    user_interactions = set(user_interactions)
    user_interactions = list(user_interactions)

    recommendations = get_recommendations(user_id, user_interactions, vectorizer)
    recommendations = recommendations[:10]

    formatted_recommendations = []
    for recommendation in recommendations:
        formatted_recommendations.append((vectorizer.get_feature_names()[recommendation[0]], recommendation[1]))

    return jsonify(formatted_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
