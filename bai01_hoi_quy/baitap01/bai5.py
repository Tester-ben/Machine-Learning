# -*- coding: utf-8 -*-
"""Bài 5: Thử hai tốc độ học khác nhau bằng Gradient Descent."""

import numpy as np
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")

x_goc = df["dien_tich"].to_numpy()

y = df["gia"].to_numpy()


# Chuẩn hóa diện tích về quanh số 0

x_tb = x_goc.mean()

x_do_lech = x_goc.std()

x = (x_goc - x_tb) / x_do_lech

def chay_gradient_descent(toc_do_hoc):

    # Khởi tạo

    w, b = 0.0, 0.0

    so_vong = 200

    n = len(x)

    print()
    print("Toc do hoc =", toc_do_hoc)
    print("Vong       w        b        MSE")

    for vong in range(1, so_vong + 1):

        # Dự đoán

        y_du_doan = w * x + b

        # Sai số

        chenh_lech = y_du_doan - y

        # Gradient

        grad_w = (2 / n) * (chenh_lech * x).sum()

        grad_b = (2 / n) * chenh_lech.sum()

        # Cập nhật w,b

        w -= toc_do_hoc * grad_w

        b -= toc_do_hoc * grad_b

        if vong in (1, 2, 5, 10, 25, 50, 100, 200):

            mse = (((w * x + b) - y) ** 2).mean()

            print(
                f"{vong:4d}  {w:7.4f}  {b:7.4f}  {mse:8.4f}"
            )

# Lần 1: tốc độ học 0.001

chay_gradient_descent(0.001)


# Lần 2: tốc độ học 1.02

chay_gradient_descent(1.02)