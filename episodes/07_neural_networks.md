---
title: "07 Neural Networks"
teaching: 10
exercises: 5
---

# Neural Network (MLPClassifier) with Breast Cancer Dataset

In this notebook, we will use a simple **Multi-layer Perceptron (MLP)** neural network to classify breast tumors.

## What is an MLP?

An **MLP** is a type of **feedforward neural network** consisting of one or more hidden layers. Each neuron computes a weighted sum of its inputs and passes the result through a nonlinear activation function.

MLPs are suitable for classification tasks and are trained using **backpropagation** to minimize loss.

## Step 1: Load the Breast Cancer Dataset

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=31
)

# Normalize (Standardize) features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Step 3: Train an MLPClassifier

We use `MLPClassifier` from Scikit-Learn with one hidden layer.

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42)
model.fit(X_train, y_train)
```

## Step 4: Evaluate the Model

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, classification_report, roc_curve, auc
)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

    Accuracy: 0.9824561403508771
    Precision: 0.9732142857142857
    Recall: 1.0
    F1 Score: 0.9864253393665159

    Classification Report:
                   precision    recall  f1-score   support

               0       1.00      0.95      0.98        62
               1       0.97      1.00      0.99       109

        accuracy                           0.98       171
       macro avg       0.99      0.98      0.98       171
    weighted avg       0.98      0.98      0.98       171

### What is a Confusion Matrix?

A **confusion matrix** shows how well the model distinguishes between classes:

|                 | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | True Positive (TP) | False Negative (FN)|
| Actual Negative | False Positive (FP)| True Negative (TN) |

This matrix lets us compute metrics like accuracy, precision, recall, and F1 score.

```python
import matplotlib.pyplot as plt
import seaborn as sns

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title('Neural Network Confusion Matrix')
plt.show()
```

![png](07_neural_networks/output_9_0.png)

### What is an ROC Curve?

The **ROC Curve** shows the trade-off between True Positive Rate and False Positive Rate.
**AUC** quantifies this performance.
Closer to 1.0 = better classifier.

```python
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='AUC = ' + str(round(roc_auc, 2)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network ROC Curve')
plt.legend()
plt.show()
```

![png](07_neural_networks/output_11_0.png)

