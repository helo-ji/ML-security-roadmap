
import pandas as pd


# 1. Creating a DataFrame

# Create a small table of network traffic data.
# Why it matters: A DataFrame is Pandas' main structure for working with tabular datasets.
data = {
    "duration": [2.1, 1.4, 0.8, 3.2, 1.1],
    "packet_length": [500, 1200, 800, 300, 1500],
    "protocol": ["TCP", "TCP", "UDP", "TCP", "UDP"],
    "attack": [0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# 2. Checking the shape

# Check the number of rows and columns.
# Why it matters: Shape tells us how large our dataset is.
print("\nShape:")
print(df.shape)


# 3. Checking column names

# Get the names of all columns.
# Why it matters: We need to know what information each column contains before working with the dataset.
print("\nColumns:")
print(df.columns)


# 4. Checking data types

# Check the data type of every column.
# Why it matters: Different types of data require different preprocessing approaches.
print("\nData types:")
print(df.dtypes)


# 5. Selecting a single column

# Select the packet_length or attack column.
# Why it matters: need to work with individual features.
print("\nPacket lengths:")
print(df["attack"])


# 6. Selecting multiple columns

# Select multiple columns from the DataFrame.
# Why it matters: ML models use selected features as input data.
print("\nSelected features:")
print(df[["duration", "packet_length"]])


# 7. Checking for missing values

# Check how many missing values exist in each column.
# Why it matters: Missing data can cause problems during preprocessing and model training.
print("\nMissing values:")
print(df.isnull().sum())


# 8. Summary statistics

# Generate basic statistics for numerical columns.
# Why it matters: Summary statistics help us understand the distribution and scale of the dataset.
print("\nSummary statistics:")
print(df.describe())


# 9. Checking the target distribution

# Count how many samples belong to each attack class.
# Why it matters: This helps us detect class imbalance.
print("\nTarget distribution:")
print(df["attack"].value_counts())


# 10. Checking unique values

# Find the different protocols present in the dataset.
# Why it matters: Categorical values may need to be encoded before they can be used by an ML model.
print("\nProtocols:")
print(df["protocol"].unique())


# 11. Basic dataset information

# Display a compact overview of the DataFrame.
# Why it matters: This quickly shows columns, data types, non-null values, and memory usage.
print("\nDataset information:")
df.info()


"""
LAB 2 SUMMARY

Pandas is used to work with structured/tabular data.

Key concepts:

DataFrame      -> table containing rows and columns
shape          -> number of rows and columns
columns        -> names of the dataset's features
dtypes         -> data type of each column
isnull()       -> finds missing values
describe()     -> provides numerical summary statistics
value_counts() -> counts occurrences of values/classes
unique()       -> shows distinct values
info()         -> provides a quick overview of the dataset

Most importantly for Machine Learning:

Before training a model, we need to understand the dataset.

We need to know:

- How many rows and columns exist?
- What does each column represent?
- Which columns are numerical?
- Which columns are categorical?
- Are there missing values?
- Are there duplicates?
- What does the target/class distribution look like?

This dataset inspection process becomes part of
Exploratory Data Analysis (EDA).
"""