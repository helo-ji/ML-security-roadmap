# lab 0 - python fundamentals

# 1. variables
name = "helo-ji"
age = 24
python_installed = True

print("Name:", name)
print("Age:", age)
print("Python installed:", python_installed)


# 2. lists
technologies = [
    "Python",
    "Machine learning",
    "Cybersecurity"
]

print("\nTechnologies:")
for technology in technologies:
    print("-", technology)


# 3. dictionaries
model = {
    "name": "Intrusion Detector",
    "features": ["packet_length", "protocol", "duration"],
    "trained": True,
    "metrics": {
        "accuracy": 0.95,
        "precision": 0.91,
        "recall": 0.89
    }

}

print("\nModel:")
print(model)


# accessing dictionary values
print("\nModel name:")
print(model["name"])

print("\nModel features:")
print(model["features"])

print("\nAccuracy:")
print(model["metrics"]["accuracy"])


# 4. loops
print("\nMetrics:")

for metric, value in model["metrics"].items():
    print(metric, "=", value)


# 5. functions
def calculate_error(actual, predicted):
    return actual - predicted


error = calculate_error(10, 8)

print("\nPrediction error:")
print(error)


# 6. conditions
def evaluate_prediction(actual, predicted):

    if actual == predicted:
        return "Correct prediction"

    else:
        return "Incorrect prediction"


print("\nPrediction evaluation:")

result = evaluate_prediction(1, 1)
print(result)

result = evaluate_prediction(1, 0)
print(result)

# all
prediction_results = [
    {"actual": 1, "predicted": 1},
    {"actual": 1, "predicted": 0},
    {"actual": 0, "predicted": 0},
    {"actual": 0, "predicted": 1}
]

print("\nPrediction results:")

for result in prediction_results:

    actual = result["actual"]
    predicted = result["predicted"]

    error = calculate_error(actual, predicted)
    evaluation = evaluate_prediction(actual, predicted)

    print(
        "Actual:", actual,
        "| Predicted:", predicted,
        "| Error:", error,
        "|", evaluation
    )