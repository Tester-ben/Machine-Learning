# -*- coding: utf-8 -*-
"""Bai tap 3: Du doan cho mot ban cu the."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# Doc du lieu
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)

w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])

print(f"He so w = {w:.6f}")
print(f"He so b = {b:.6f}")
print()


def du_doan(gio):
    # Tinh z = wx + b
    z = w * gio + b

    # Tinh xac suat bang sigmoid
    xac_suat = 1 / (1 + np.exp(-z))

    # Gan nhan theo nguong 0.5
    if xac_suat >= 0.5:
        nhan = 1
    else:
        nhan = 0

    print(f"So gio on: {gio}")
    print(f"z = {z:.4f}")
    print(f"Xac suat qua mon = {xac_suat:.4f}")
    print(f"Nhan du doan = {nhan}")
    print()


for gio in [3, 8, 12.89, 18, 26]:
    du_doan(gio)