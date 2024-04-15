import pickle

# Load the model
filename = 'model.sav'
vectorizer = pickle.load(open(filename, 'rb'))
