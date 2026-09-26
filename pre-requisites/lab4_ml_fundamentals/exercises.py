"""
LAB 4 SUMMARY

Machine Learning is the process of using data to learn
patterns that can be used to make predictions or decisions.

Key concepts:

Feature       -> input variable used by the model
Target        -> value the model is trying to predict
X             -> conventionally represents the input/features
y             -> conventionally represents the target/label

Training data -> data used to teach the model
Testing data  -> data used to evaluate the model

fit()         -> trains the model using the training data
predict()     -> uses the trained model to make predictions

The basic Machine Learning workflow is:

1. Load or create the dataset
2. Separate features from the target
3. Split the data into training and testing sets
4. Create a Machine Learning model
5. Train the model using fit()
6. Make predictions using predict()
7. Evaluate the predictions

Important:
We should NOT evaluate a model only on the same data
that was used to train it.

A model needs to be tested on data it has not seen before.

This helps us determine whether the model has learned
a useful pattern rather than simply memorizing the training data.

This concept becomes extremely important later in
ML Security because understanding how models learn,
generalize, and fail is necessary before studying
attacks against Machine Learning systems.
"""

import pandas as pd

from sklearn.model_selection import train_test_split #
# to divide datasets
from sklearn.linear_model import LinearRegression
# first ml model
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "exam_score": [52, 55, 60, 65, 70, 74, 78, 83, 88, 92]
}


df = pd.DataFrame(data)
print(df)

# understand x and y
X = df[["study_hours"]]
y = df["exam_score"]
# but why two brackets, one to produce dataframe, second to series, NOT  X = df["study_hours"]

# split x and y into training and testing portions, keep 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#X_train : feature values used for training
#X_test  : feature values reserved for testing

#y_train : correct answers for training
#y_test  : correct answers for testing
# keep random state same if you want reproducible results

# create model, imagine an empty learner
model = LinearRegression()

# train the model
model.fit(X_train, y_train)
#X_train : the inputs
#y_train : the correct answers

# make predictions
predictions = model.predict(X_test)

print("Actual values:")
print(y_test)

print("\nPredicted values:")
print(predictions)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)