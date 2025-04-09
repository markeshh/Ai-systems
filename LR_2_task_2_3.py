# === ІМПОРТ БІБЛІОТЕК ===
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import scatter_matrix
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# === КРОК 1. ЗАВАНТАЖЕННЯ ДАНИХ ===
iris = load_iris()
X = iris.data
y = iris.target

print("Назви ознак:", iris.feature_names)
print("Назви класів:", iris.target_names)
print("Перші 5 прикладів X:\n", X[:5])
print("Мітки y:", y[:5])

# === КРОК 2. ВІЗУАЛІЗАЦІЯ ДАНИХ ===
# Завантаження як DataFrame для побудови графіків
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris_df = read_csv(url, names=names)

# 📊 Діаграма розмаху
iris_df.iloc[:, 0:4].plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.suptitle("Діаграма розмаху ознак Iris")
plt.tight_layout()
plt.show()

# 📊 Гістограми ознак
iris_df.hist()
plt.suptitle("Гістограми ознак Iris")
plt.tight_layout()
plt.show()

# 📊 Матриця діаграм розсіювання
scatter_matrix(iris_df.iloc[:, 0:4])
plt.suptitle("Scatter Matrix ознак Iris")
plt.tight_layout()
plt.show()

# === КРОК 3. РОЗБИТТЯ ДАНИХ ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)

# === КРОК 4. КЛАСИФІКАЦІЯ РІЗНИМИ МЕТОДАМИ ===
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

print("\nОцінка точності моделей (10-fold cross-validation):")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ± {cv_results.std():.4f}")

# 📊 Побудова діаграми порівняння моделей
plt.boxplot(results, labels=names)
plt.title("Порівняння алгоритмів класифікації (Accuracy)")
plt.grid(True)
plt.show()

# === КРОК 5. КОНТРОЛЬНЕ ТЕСТУВАННЯ НА SVM ===
model = SVC(gamma='auto')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nОцінка моделі SVM на тестовій вибірці:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# === КРОК 6. ПРОГНОЗ ДЛЯ НОВОЇ КВІТКИ ===
X_new = np.array([[5.5, 4.1, 1.5, 0.3]])
prediction = model.predict(X_new)

print("\nПрогноз для нової квітки:")
print("Клас:", prediction[0])
print("Це сорт:", iris.target_names[prediction[0]])
