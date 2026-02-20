
# iris_knn_classification.py
# -----------------------------------------------------------
# K-Nearest Neighbors (KNN) Classification on Iris Dataset
# -----------------------------------------------------------

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix
import mglearn

# -----------------------------------------------------------
# 1. Load the Iris dataset
# -----------------------------------------------------------
iris_dataset = load_iris()

# -----------------------------------------------------------
# 2. Split dataset into training and testing sets
# -----------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    iris_dataset['data'],
    iris_dataset['target'],
    random_state=0
)

# -----------------------------------------------------------
# 3. Convert training data to Pandas DataFrame for visualization
# -----------------------------------------------------------
iris_dataframe = pd.DataFrame(
    x_train,
    columns=iris_dataset.feature_names
)

# Display basic dataset information
print("Training feature shape:", x_train.shape)
print("Testing feature shape:", x_test.shape)
print("Training label shape:", y_train.shape)
print("Testing label shape:", y_test.shape)

# -----------------------------------------------------------
# 4. Visualize data using scatter matrix
# -----------------------------------------------------------
scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(12, 12),
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=0.8,
    cmap=mglearn.cm3
)

plt.show()

# -----------------------------------------------------------
# 5. Create and train KNN model
# -----------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=1)

# Train model using training data
knn.fit(x_train, y_train)

# -----------------------------------------------------------
# 6. Predict class for a new sample input
# -----------------------------------------------------------
x_new = np.array([[5, 2.9, 1, 0.2]])
print("New sample shape:", x_new.shape)

prediction = knn.predict(x_new)
print("Predicted class index:", prediction)

print("Predicted flower name:",
      iris_dataset['target_names'][prediction][0])

# -----------------------------------------------------------
# 7. Evaluate model performance
# -----------------------------------------------------------
y_pred = knn.predict(x_test)

print("Test set predictions:", y_pred)

# Manual accuracy calculation
manual_accuracy = np.mean(y_pred == y_test)
print("Manual accuracy:", manual_accuracy)

# Built-in accuracy calculation
model_accuracy = knn.score(x_test, y_test)
print("Model accuracy:", model_accuracy)

