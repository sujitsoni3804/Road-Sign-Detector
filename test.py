import pickle

with open("Scikit_learn/model.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))