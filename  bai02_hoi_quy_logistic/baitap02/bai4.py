# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# Doc du lieu
df = pd.read_csv("data/sinh_vien.csv")

X = df[["gio_on"]]
y = df["qua_mon"]

# Chia du lieu giong trong tai lieu
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

# Huan luyen mo hinh
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)

# Du doan tren tap kiem tra
y_pred = mo_hinh.predict(X_test)

# Lay 4 gia tri TN, FP, FN, TP
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("MA TRAN NHAM LAN")
print("-----------------------------")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")
print()

# Tu tinh 4 thuoc do

# Accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

# Precision
precision = tp / (tp + fp)

# Recall
recall = tp / (tp + fn)

# F1-score
f1 = 2 * precision * recall / (precision + recall)

print("KET QUA TU TINH")
print("-----------------------------")
print(f"Accuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1        = {f1:.4f}")
print()

print("TU TINH THEO CONG THUC")
print("-----------------------------")
print(
    f"Accuracy  = ({tp} + {tn}) / "
    f"({tp} + {tn} + {fp} + {fn}) = {accuracy:.4f}"
)

print(
    f"Precision = {tp} / "
    f"({tp} + {fp}) = {precision:.4f}"
)

print(
    f"Recall    = {tp} / "
    f"({tp} + {fn}) = {recall:.4f}"
)

print(
    f"F1        = 2 * {precision:.4f} * {recall:.4f} / "
    f"({precision:.4f} + {recall:.4f}) = {f1:.4f}"
)