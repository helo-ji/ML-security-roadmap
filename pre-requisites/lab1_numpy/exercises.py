# lab 1 numpy
import numpy as np

# 1. creating numy arrays

# Create a NumPy array containing numerical values.
# Why it matters: NumPy arrays are the basic structure we use to efficiently store and manipulate numerical ML data.
features = np.array([
    10,
    20,
    30,
    40,
    50
])

print("Features:")
print(features)


# 2. checking the shape

# Check the dimensions/size of the array
# Why it matters: In ML, shape tells us how many samples and features our data contains
print("\nShape:")
print(features.shape)


# 3. indexing

# Get the first value (index 0).
# Why it matters: Indexing lets us access individual pieces of data.
print("\nFirst value:")
print(features[0])


# Get the third value (index 2).
# Remember: Python starts counting at 0
print("\nThird value:")
print(features[2])


# Get the last value.
# Negative indexing lets us count backwards from the end.
print("\nLast value:")
print(features[-1])


# 4. slicing

# Start at the beginning and stop before index 3.
# Result: [10, 20, 30]
# Why it matters: Slicing lets us select a section of our data.
print("\nFirst three values:")
print(features[:3])


# Start at index 2 and continue to the end.
# Result: [30, 40, 50]
print("\nValues from index 2 onward:")
print(features[2:])


# Start at index 1 and stop before index 4.
# Result: [20, 30, 40]
# Remember the stop index is NOT included.
print("\nMiddle values:")
print(features[1:4])


# 5. Vectorized operations

# Add 10 to every value in the array.
# Why it matters: NumPy can perform mathematical operations across entire datasets without manually looping through them.
print("\nAdd 10:")
print(features + 10)


# Multiply every value by 2.
print("\nMultiply by 2:")
print(features * 2)


# Divide every value by 10.
print("\nDivide by 10:")
print(features / 10)


# 6. basic statistics

# Calculate the average of all values.
# Why it matters: The mean is commonly used when analyzing
# and preprocessing datasets.
print("\nMean:")
print(features.mean())


# Calculate the standard deviation.
# Why it matters: Standard deviation tells us how spread out the values are around the average.
print("\nStandard deviation:")
print(features.std())


# 7. two-dimensional arrays

# Create a 2D array with rows and columns.
# Think of this like a small dataset/table
# Why it matters: ML datasets are commonly represented as
# rows = samples and columns = features.
data = np.array([
    [10, 100],
    [20, 200],
    [30, 300],
    [40, 400]
])

print("\n2D dataset:")
print(data)


# Check the shape of the 2D array.
# Result: (4, 2)
# Meaning: 4 rows and 2 columns.
print("\nDataset shape:")
print(data.shape)


# 8. accessing rows and columns

# Get the first row (index 0).
# Why it matters: We often need to access individual samples from a dataset
print("\nFirst row:")
print(data[0])


# Get every row from column 0.
# ":" means all rows.
# "0" means column 0.
# Why it matters: Selecting columns is essential when working with features in ML datasets.
print("\nFirst column:")
print(data[:, 0])


# Get every row from column 1.
print("\nSecond column:")
print(data[:, 1])


# 9. working with realistic numerical data

# Create an array representing network packet lengths.
# Why it matters: This gives us an example closer to the numerical data we'll encounter in the intrusion detection project.
packet_lengths = np.array([
    500,
    1200,
    800,
    300,
    1500
])

print("\nPacket lengths:")
print(packet_lengths)


# Calculate the average packet length.
# Why it matters: Summary statistics help us understand the characteristics of a dataset.
print("\nAverage packet length:")
print(packet_lengths.mean())


# Find the largest packet length.
print("\nLargest packet:")
print(packet_lengths.max())


# Find the smallest packet length.
print("\nSmallest packet:")
print(packet_lengths.min())

"""
LAB 1 SUMMARY

NumPy provides efficient arrays and mathematical operations
for working with numerical data.

Key concepts:

array      -> stores numerical data
shape      -> tells us the dimensions of the data
indexing   -> accesses individual values
slicing    -> accesses sections of data
vectorized -> performs operations across entire arrays
mean       -> calculates the average
std        -> measures how spread out values are
min/max    -> finds the smallest/largest values

Most importantly for Machine Learning:

Rows represent samples.
Columns represent features.

Example:

X.shape = (1000, 20)

This means:
1000 samples
20 features

This way of thinking about data will be important
throughout the rest of the roadmap.
"""