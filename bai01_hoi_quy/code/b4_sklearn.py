# -*- coding: utf-8 -*-
"""Bước 4: làm lại đúng việc đó bằng scikit-learn."""


import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("data/gia_nha.csv")


# X là bảng 2 chiều, y là 1 chiều
X = df[["dien_tich"]]
y = df["gia"]


mo_hinh = LinearRegression()

mo_hinh.fit(X, y)


print("He so goc w =", round(float(mo_hinh.coef_[0]), 6))
print("He so chan b =", round(float(mo_hinh.intercept_), 6))

print()

print("Ket qua tinh tay o buoc 3: w = 0.078367, b = 0.401752")

print()


can_moi = pd.DataFrame({"dien_tich":[80.0, 100.0]})

du_doan = mo_hinh.predict(can_moi)


for dt, gia in zip(can_moi["dien_tich"], du_doan):
    print(f"Can {dt:.0f} m2 -> du doan {gia:.3f} ty dong")