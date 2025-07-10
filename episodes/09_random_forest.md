---
title: "09 Random Forest"
teaching: 10
exercises: 5
---

# Random Forest Classifier with Breast Cancer Dataset

This notebook demonstrates the use of **Random Forest Classifier** for classifying tumors in the Breast Cancer dataset.

## What is a Random Forest?

Random Forest is a powerful ensemble learning method for classification. It builds multiple **decision trees** and combines their predictions for improved accuracy and robustness.

Each tree is trained on a random subset of the data and features, reducing overfitting and improving generalization.

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

## Step 3: Train an SVM Model

We use the `SVC` class from `sklearn.svm` with default kernel (RBF).

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
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

    Accuracy: 0.9590643274853801
    Precision: 0.9636363636363636
    Recall: 0.9724770642201835
    F1 Score: 0.9680365296803652

    Classification Report:
                   precision    recall  f1-score   support

               0       0.95      0.94      0.94        62
               1       0.96      0.97      0.97       109

        accuracy                           0.96       171
       macro avg       0.96      0.95      0.96       171
    weighted avg       0.96      0.96      0.96       171

### What is a Confusion Matrix?

A **confusion matrix** is a summary of prediction results:

|                 | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | True Positive (TP) | False Negative (FN)|
| Actual Negative | False Positive (FP)| True Negative (TN) |

- Accuracy, Precision, Recall, F1 Score are all derived from this table.

```python
import matplotlib.pyplot as plt
import seaborn as sns

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title('SVM Confusion Matrix')
plt.show()
```

![png](09_random_forest/output_9_0.png)

### What is an ROC Curve?

The **ROC Curve** shows the trade-off between True Positive Rate (Recall) and False Positive Rate.
The **AUC (Area Under Curve)** summarizes the performance into a single number.

```python
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='AUC = ' + str(round(roc_auc, 2)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend()
plt.show()
```

![png](09_random_forest/output_11_0.png)

