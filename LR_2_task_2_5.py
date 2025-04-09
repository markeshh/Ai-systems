import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# Завантаження датасету Iris
iris = load_iris()
X, y = iris.data, iris.target

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Ініціалізація та тренування моделі RidgeClassifier
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)

# Прогноз
y_pred = clf.predict(X_test)

# Обчислення метрик
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, average='weighted'), 4))
print("Recall:", round(recall_score(y_test, y_pred, average='weighted'), 4))
print("F1 Score:", round(f1_score(y_test, y_pred, average='weighted'), 4))
print("Cohen's Kappa:", round(cohen_kappa_score(y_test, y_pred), 4))
print("Matthews Correlation Coefficient:", round(matthews_corrcoef(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Матриця плутанини
mat = confusion_matrix(y_test, y_pred)

# Побудова теплокарти
plt.figure(figsize=(6, 5))
sns.set()
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.tight_layout()

# Збереження графіка у файл
plt.savefig("Confusion.jpg", dpi=300)
plt.show()
