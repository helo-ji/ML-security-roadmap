import pandas as pd
import matplotlib.pyplot as plt

# 1. data creation
data = {
    "age":         [18, 21, 25, 30, 35, 40, 45, 50, 55, 60],
    "study_hours": [2, 3, 4, 5, 6, 7, 6, 8, 9, 10],
    "exam_score":  [55, 58, 62, 68, 72, 78, 75, 84, 88, 92]
}


# convert to dataframe and shorten it
df = pd.DataFrame(data)


# 2. data inspection

# display the dataset
print("Original dataset")
print(df)
print("end")

# display the first five rows
print("\nFirst five rows")
print(df.head())
print("\nend")


# info about the dataframe itself
print("\ninfo")
print(df.info())
print("end")


# columns
print("\ndescription")
print(df.describe())
print("end")


# 3. histogram
# select exam_score, create histogram into approx 5 groups/bin
plt.hist(df["exam_score"], bins=5)

plt.xlabel("Exam Score (x)")
plt.ylabel("Number of Students (y)")
plt.title("Distribution of Exam Scores")

plt.show()

# scatter plot
# two pieces of data, first is x asis, second is y
plt.scatter(df["study_hours"], df["exam_score"])

plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")

plt.show()

# correlation
# if close to 1, strong postiive relationship
# close to -1, strong negative relationship
# 0 means little or no relationship
correlation = df.corr()

print("\nCorrelation Matrix")
print(correlation)


# visualize the correlation matrix
plt.imshow(correlation)

plt.colorbar()

plt.xticks(
    # range(3)
    range(len(correlation.columns)),
    correlation.columns
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.show()


# put outlier IMPORTANT need to understand
# initial 10 rows
# next index is 10 as its 0, 1, 2 ... 9
df.loc[len(df)] = [100, 1, 95] # df.loc[10] = [100,1, 95]
# means 
# age = 100
# study_hours = 1
# exam_score = 95

print("\nDataset After Adding Unusual Observation")
print(df)


# Plot the data again
plt.scatter(df["study_hours"], df["exam_score"])

plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score — With Unusual Observation")

plt.show()

# what if there is missing data
df.loc[3, "exam_score"] = None

print("\nDataset After Introducing Missing Value")
print(df)


# Check for missing values
print("\nMissing Values")
print(df.isnull().sum())

# summary
print("\nFinal Dataset Information")
df.info()


print("\nFinal Statistical Summary")
print(df.describe())

"""
LAB 3 SUMMARY

Exploratory Data Analysis (EDA) is the process of inspecting,
understanding, and visualizing a dataset before building
a Machine Learning model.

Key concepts:

EDA            -> Exploratory Data Analysis
histogram      -> shows the distribution of numerical data
scatter plot   -> shows the relationship between two variables
correlation    -> measures how strongly two numerical variables are related
outlier        -> an unusual observation compared to the rest
isnull()       -> finds missing values
corr()         -> calculates correlations between numerical columns
describe()     -> provides numerical summary statistics
info()         -> provides a quick overview of the dataset

In this lab, we use:

age            -> student's age
study_hours    -> number of hours studied
exam_score     -> student's exam score

Most importantly for Machine Learning:

Before training a model, we need to understand the data.

We need to know:

- What does the dataset look like?
- How are the numerical values distributed?
- Are there relationships between variables?
- Are there unusual observations or outliers?
- Are there missing values?
- Are there patterns that might affect the model?

Visualization helps us see patterns that may not be obvious
when looking at raw data.

An important concept is:

Correlation does NOT automatically mean causation.

An unusual observation should also not automatically be removed.

We should investigate whether an outlier is:

- A legitimate observation
- A data-entry error
- A measurement error
- A special case

Missing data must also be identified and handled appropriately
before using the dataset for Machine Learning.

This EDA process becomes especially important later in
ML Security, where unusual data, anomalies, distribution
changes, and potentially manipulated data may need to be detected.
"""