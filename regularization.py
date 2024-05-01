"""
The target is to create a classification model using L2 regularization on the
breast_cancer dataset. It contains features computed from a digitized image of
a fine needle aspirate (FNA) of a breast mass.
The task associated with this dataset is to classify the breast mass as
malignant (cancerous) or benign (non-cancerous) based on the extracted
features.
"""
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix


# Set environment
SEED = 42
np.random.seed(SEED)


# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target


# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)  # noqa

# Creating a Regression Logistic Model
model = LogisticRegression(penalty='l2', C=1, random_state=SEED)

# Fit the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate F1 Score and confusion matrix
f1 = f1_score(y_test, y_pred)
print(f'F1-score: {f1}')
print(f'Confussion matrix: \n{confusion_matrix(y_test, y_pred)}')
