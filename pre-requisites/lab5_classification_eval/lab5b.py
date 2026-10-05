"""
LAB 5B to check evaluation

In this experiment, we intentionally create incorrect predictions so we can see how the confusion matrix and evaluation metrics change.

The goal is not to build a good model.
The goal is to understand:
- True Positives
- True Negatives
- False Positives
- False Negatives
- Accuracy
- Precision
- Recall
- F1 Score
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# actual answers from test set
y_test = [0, 0, 0, 0, 0,
          0, 0, 0, 0, 0,
          1, 1, 1, 1, 1,
          1, 1, 1, 1,
          2, 2, 2, 2, 2,
          2, 2, 2, 2, 2,
          2]


# Pretend these are the model's predictions
y_pred = [0, 0, 0, 0, 0,
          0, 0, 1, 1, 1,
          1, 1, 1, 1, 1,
          1, 1, 1, 2,
          2, 2, 2, 2, 2,
          2, 2, 2, 2, 2,
          2]


print("Actual:", y_test)
print("Predicted:", y_pred)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


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