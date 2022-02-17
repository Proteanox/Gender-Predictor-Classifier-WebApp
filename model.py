import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('weight-height.csv')

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

X = df[['Height', 'Weight']]
Y = df.Gender

clf = RandomForestClassifier()
clf.fit(X, Y)

y_pred = clf.predict(X)

print(y_pred)

print(accuracy_score(Y, y_pred))

# saving the model

pickle_out = open("classifier.pkl", mode="wb")
pickle.dump(clf, pickle_out)
pickle_out.close()
