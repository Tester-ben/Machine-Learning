# -*- coding: utf-8 -*-
"""Bài 3: Đổi biến đầu vào sang tuổi nhà."""


import numpy as np
import pandas as pd


# Đọc dữ liệu

df = pd.read_csv("data/gia_nha.csv")


# Lấy biến đầu vào và biến mục tiêu
# Đổi từ dien_tich sang tuoi_nha

x = df["tuoi_nha"].to_numpy()

y = df["gia"].to_numpy()


# Trung bình

x_tb = x.mean()

y_tb = y.mean()


# Công thức bình phương tối thiểu

tu_so = ((x - x_tb) * (y - y_tb)).sum()

mau_so = ((x - x_tb) ** 2).sum()


# Tính hệ số góc và hệ số chặn

w = tu_so / mau_so

b = y_tb - w * x_tb


print(f"Trung binh tuoi nha : {x_tb:.4f}")

print(f"Trung binh gia      : {y_tb:.4f}")

print(f"Tu so               : {tu_so:.4f}")

print(f"Mau so              : {mau_so:.4f}")

print()


print(f"He so goc      w = {w:.6f}")

print(f"He so chan     b = {b:.6f}")

print()

print(
    f"Mo hinh: gia = {w:.4f} * tuoi_nha + {b:.4f}"
)

print()

# Tính MSE trên toàn bộ 60 căn

y_du_doan = w * x + b

mse = ((y - y_du_doan) ** 2).mean()

print(
    f"MSE tren toan bo 60 can: {mse:.4f}"
)

print()

# Dự đoán căn nhà tuổi 10 năm

print(
    f"Can 10 nam tuoi duoc du doan: {w * 10 + b:.3f} ty dong"
)

print()

print(
    "Nhan xet: w mang dau am, nghia la khi tuoi nha tang "
    "thi gia nha co xu huong giam."
)

print(
    "Tuy nhien cac diem du lieu khong bam sat hoan toan vao "
    "mot duong thang nen chi nhin vao dau cua w chua du de "
    "ket luan moi quan he giua tuoi nha va gia rat manh."
)