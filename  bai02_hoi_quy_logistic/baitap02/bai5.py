# -*- coding: utf-8 -*-
"""Bai tap 5: Do nguong tot nhat theo F1."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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
p = mo_hinh.predict_proba(X_test)[:, 1]
nguong_tot_nhat = 0
f1_tot_nhat = -1

print("Nguong\tF1")
print("----------------")

# Thu cac nguong tu 0.05 den 0.95
for nguong in np.arange(0.05, 1.0, 0.05):

    y_pred = (p >= nguong).astype(int)

    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"{nguong:.2f}\t{f1:.4f}")

    if f1 > f1_tot_nhat:
        f1_tot_nhat = f1
        nguong_tot_nhat = nguong


print()
print("KET QUA")
print("----------------")
print(f"Nguong cho F1 cao nhat = {nguong_tot_nhat:.2f}")
print(f"F1 cao nhat = {f1_tot_nhat:.4f}")

# Nhan xet
if abs(nguong_tot_nhat - 0.5) < 1e-9:
    print("Nhan xet: Nguong tot nhat theo F1 dung bang 0.5.")
else:
    print("Nhan xet: Nguong tot nhat theo F1 khong bang 0.5.")