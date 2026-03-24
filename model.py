import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# load dataset
data = pd.read_csv("disease.csv")

print("Dataset loaded")

# features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split done")

# create model
model = DecisionTreeClassifier()

# train model
model.fit(X_train, y_train)

print("Model trained")

# test prediction
sample = X.iloc[0]

prediction = model.predict([sample])

print("Prediction:", prediction)