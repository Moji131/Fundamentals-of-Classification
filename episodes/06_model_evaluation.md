---
title: "06 Model Evaluation"
teaching: 10
exercises: 5
---

# Model Evaluation: Comparing Logistic Regression and SVM

We compare two classifiers:
- **Logistic Regression**
- **SVM**

Using metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC-AUC.

## Step 1: Load the Data

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

## Step 2: Train the Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(X_train, y_train)

svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

```


## Step 3: Evaluation Metrics

### What is Accuracy?

**Accuracy** is the proportion of correct predictions:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Can be misleading if classes are imbalanced.

```python
from sklearn.metrics import accuracy_score

lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
svm_acc = accuracy_score(y_test, svm_model.predict(X_test))

print(f"Logistic egression Accuracy: {lr_acc:.2f}")
print(f"SVM Accuracy: {svm_acc:.2f}")

```

    Logistic egression Accuracy: 0.98
    SVM Accuracy: 0.98

### What are Precision, Recall and F1 Score?

- **Precision**: $\frac{TP}{TP + FP}$  
- **Recall**: $\frac{TP}{TP + FN}$  
- **F1 Score**: Harmonic mean of precision and recall  

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

```python
from sklearn.metrics import precision_score, recall_score, f1_score

for name, model in [("SVM", svm_model), ("Neural Net", lr_model)]:
    y_pred = model.predict(X_test)
    print(f"\n{name} Metrics:")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
```

    SVM Metrics:
    Precision: 0.97
    Recall: 0.99
    F1 Score: 0.98

    Neural Net Metrics:
    Precision: 0.98
    Recall: 0.98
    F1 Score: 0.98

### What is a Confusion Matrix?

A **confusion matrix** shows the breakdown of correct and incorrect classifications.

|                 | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | True Positive (TP) | False Negative (FN)|
| Actual Negative | False Positive (FP)| True Negative (TN) |

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test, ax=axs[0])
axs[0].set_title("SVM Confusion Matrix")
ConfusionMatrixDisplay.from_estimator(lr_model, X_test, y_test, ax=axs[1])
axs[1].set_title("Neural Network Confusion Matrix")
plt.tight_layout()
plt.show()
```

![png](06_model_evaluation/output_11_0.png)

### What is the ROC Curve?

ROC = Receiver Operating Characteristic Curve

- Plots TPR vs FPR
- **AUC** = Area Under the ROC Curve
Closer to 1 = better model.

```python
from sklearn.metrics import roc_curve, auc

svm_probs = svm_model.predict_proba(X_test)[:, 1]
nn_probs = lr_model.predict_proba(X_test)[:, 1]

svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_probs)
svm_auc = auc(svm_fpr, svm_tpr)
nn_auc = auc(nn_fpr, nn_tpr)

plt.figure(figsize=(8, 6))
plt.plot(svm_fpr, svm_tpr, label=f"SVM (AUC = {svm_auc:.2f})")
plt.plot(nn_fpr, nn_tpr, label=f"Logistic Regression (AUC = {nn_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

![png](06_model_evaluation/output_13_0.png)

## Conclusion

Both models perform well, but:

- **Neural Net** may achieve higher recall
- **SVM** may offer higher precision

Evaluation metrics guide us to choose the best model for our real-world use case.
