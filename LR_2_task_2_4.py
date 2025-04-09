import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

input_file = 'income_data.txt'
X_raw = []
count_class1 = 0
count_class2 = 0
max_datapoints = 5000  # Зменшено для пришвидшення

with open(input_file, 'r') as f:
    for line in f:
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X_raw.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X_raw.append(data)
            count_class2 += 1
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

X_raw = np.array(X_raw)


X_encoded = np.empty(X_raw.shape)
label_encoders = []
for i in range(X_raw.shape[1]):
    if X_raw[0, i].isdigit():
        X_encoded[:, i] = X_raw[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X_raw[:, i])
        label_encoders.append(encoder)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)   


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)


models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
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


plt.boxplot(results, labels=names)
plt.title("Порівняння алгоритмів класифікації (Accuracy)")
plt.grid(True)
plt.show()


model = SVC(gamma='auto')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nОцінка SVM на тестовій вибірці:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))