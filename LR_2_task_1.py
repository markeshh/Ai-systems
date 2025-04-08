import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Вхідний файл
input_file = 'income_data.txt'

# Змінні
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

# Зчитування даних
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

# Кодування категоріальних змінних
label_encoders = []
X_encoded = np.empty(X.shape)
for i in range(X.shape[1]):
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoders.append(encoder)

# Розділення на X (ознаки) та y (мітки)
X_features = X_encoded[:, :-1].astype(int)
y_labels = X_encoded[:, -1].astype(int)

# Розбивка на train/test
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=5)

# Створення та навчання класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
classifier.fit(X_train, y_train)

# Прогнозування
y_pred = classifier.predict(X_test)

# Оцінка моделі
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {acc:.2%}")
print(f"Precision: {prec:.2%}")
print(f"Recall:    {rec:.2%}")
print(f"F1 score:  {f1:.2%}")

# Тестова точка
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

# Кодування тестової точки
input_encoded = []
label_index = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_encoded.append(int(item))
    else:
        encoder = label_encoders[label_index]
        input_encoded.append(int(encoder.transform([item])[0]))
        label_index += 1

input_encoded = np.array(input_encoded).reshape(1, -1)

# Передбачення класу
predicted_class = classifier.predict(input_encoded)
output = label_encoders[-1].inverse_transform(predicted_class)
print(f"Predicted class for input data: {output[0]}")
