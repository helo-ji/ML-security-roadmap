"""
Lab 5 summary

In this lab, we learn how to evaluate a classification model

We will:
1. Load a dataset
2. Separate features (X) and target (y)
3. Split the data into training and testing sets
4. Train a classification model
5. Make predictions on the test set
6. Evaluate the predictions using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix

Why this matters:
Training a model is only part of machine learning.
We also need to measure how well the model performs on data
it has not seen before.

These evaluation metrics are especially important in
security-related ML because different types of mistakes
can have very different consequences.
"""

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# load dataset
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

print(X.head())
print(y.head())

# split train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)
print("Predictions:")
print(y_pred)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nPrecision:", precision)


recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

print("Recall:", recall)


f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("F1 Score:", f1)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""
Precision cares about the predictions you made
Recall cares about the actual examples you were supposed to find.

                MODEL PREDICTIONS
                       │
                       ▼
                CONFUSION MATRIX
                       │
             ┌─────────┼─────────┐
             │         │         │
             ▼         ▼         ▼
          Accuracy  Precision  Recall
                                  │
                                  ▼
                               F1 Score
"""