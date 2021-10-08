import pickle

with open('problems1.pkl', 'rb') as f:
    data1 = pickle.load(f)

with open('problems2.pkl', 'rb') as f:
    data2 = pickle.load(f)

data = {**data1, **data2}

with open('problems.pkl', 'wb') as f:
    pickle.dump(data, f)