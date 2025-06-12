---
title: "Model Evaluation: Comparing SVM and Neural Network"
teaching: 25
exercises: 15
---

# Model Evaluation: Comparing SVM and Neural Network

We compare two classifiers:
- **SVM**
- **Neural Network**

Using metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC-AUC.

## Step 1: Load the Data

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## Step 2: Train the Models

```python
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

svm_model = SVC(probability=True)
nn_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42)

svm_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)
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

svm_acc = accuracy_score(y_test, svm_model.predict(X_test))
nn_acc = accuracy_score(y_test, nn_model.predict(X_test))

print(f"SVM Accuracy: {svm_acc:.2f}")
print(f"Neural Network Accuracy: {nn_acc:.2f}")
```

### What are Precision, Recall and F1 Score?

- **Precision**: $\frac{TP}{TP + FP}$  
- **Recall**: $\frac{TP}{TP + FN}$  
- **F1 Score**: Harmonic mean of precision and recall  

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

```python
from sklearn.metrics import precision_score, recall_score, f1_score

for name, model in [("SVM", svm_model), ("Neural Net", nn_model)]:
    y_pred = model.predict(X_test)
    print(f"\n{name} Metrics:")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
```

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
ConfusionMatrixDisplay.from_estimator(nn_model, X_test, y_test, ax=axs[1])
axs[1].set_title("Neural Network Confusion Matrix")
plt.tight_layout()
plt.show()
```

### What is the ROC Curve?

ROC = Receiver Operating Characteristic Curve

- Plots TPR vs FPR
- **AUC** = Area Under the ROC Curve
Closer to 1 = better model.

```python
from sklearn.metrics import roc_curve, auc

svm_probs = svm_model.predict_proba(X_test)[:, 1]
nn_probs = nn_model.predict_proba(X_test)[:, 1]

svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_probs)
svm_auc = auc(svm_fpr, svm_tpr)
nn_auc = auc(nn_fpr, nn_tpr)

plt.figure(figsize=(8, 6))
plt.plot(svm_fpr, svm_tpr, label=f"SVM (AUC = {svm_auc:.2f})")
plt.plot(nn_fpr, nn_tpr, label=f"Neural Net (AUC = {nn_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

## Conclusion

Both models perform well, but:

- **Neural Net** may achieve higher recall
- **SVM** may offer higher precision

Evaluation metrics guide us to choose the best model for our real-world use case.

