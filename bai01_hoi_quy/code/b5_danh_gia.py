# -*- coding: utf-8 -*-
"""Bước 5: chia dữ liệu và chấm điểm mô hình bằng các độ đo chuẩn."""


import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



df = pd.read_csv("data/gia_nha.csv")

X = df[["dien_tich"]]
y = df["gia"]

# 80% để học, 20% để kiểm tra

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("So can de hoc   :", len(X_train))
print("So can de kiem tra:", len(X_test))

print()


mo_hinh = LinearRegression()

mo_hinh.fit(X_train, y_train)



# Dự đoán trên tập kiểm tra

y_pred = mo_hinh.predict(X_test)



# Tính các độ đo

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print(f"MAE  = {mae:.4f} ty dong")

print(f"MSE  = {mse:.4f}")

print(f"RMSE = {rmse:.4f} ty dong")

print(f"R2   = {r2:.4f}")

print()


print("Mot vai can trong tap kiem tra:")


for dt, that, du_doan in list(
    zip(
        X_test["dien_tich"],
        y_test,
        y_pred
    )
)[:5]:

    print(
        f"{dt:6.1f} m2  "
        f"that = {that:5.2f}  "
        f"du doan = {du_doan:5.2f}  "
        f"sai so = {that-du_doan:+5.2f}"
    )