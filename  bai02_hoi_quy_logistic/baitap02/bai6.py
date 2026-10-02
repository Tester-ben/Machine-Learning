# -*- coding: utf-8 -*-
"""Bai tap 6: Doi lop duong roi cham lai."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix
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

# Du doan
y_pred = mo_hinh.predict(X_test)

# Ma tran nham lan
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("MA TRAN NHAM LAN")
print("-----------------------------")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")
print()

# Lop 1 = qua mon
precision_1 = precision_score(y_test, y_pred, pos_label=1)
recall_1 = recall_score(y_test, y_pred, pos_label=1)

# Lop 0 = rot mon
precision_0 = precision_score(y_test, y_pred, pos_label=0)
recall_0 = recall_score(y_test, y_pred, pos_label=0)

print("KET QUA CHO LOP 1 - QUA MON")
print("-----------------------------")
print(f"Precision lop 1 = {precision_1:.4f}")
print(f"Recall lop 1    = {recall_1:.4f}")
print()

print("KET QUA CHO LOP 0 - ROT MON")
print("-----------------------------")
print(f"Precision lop 0 = {precision_0:.4f}")
print(f"Recall lop 0    = {recall_0:.4f}")
print()

print("NHAN XET")
print("-----------------------------")
print("Hai bo so khac nhau vi khi doi lop duong,")
print("vai tro cua TP, TN, FP va FN cung thay doi.")
print("Mo hinh va du lieu khong thay doi,")
print("chi thay doi lop nao duoc xem la lop duong.")