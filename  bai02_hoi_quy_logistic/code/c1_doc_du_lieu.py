# -*- coding: utf-8 -*-
"""Bước 1: đọc bộ dữ liệu sinh viên và xem qua một lượt."""

import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("data/sinh_vien.csv")

# Kích thước dữ liệu
print("Kich thuoc bang (so dong, so cot):", df.shape)
print()

# Xem 5 dòng đầu
print("Nam dong dau tien:")
print(df.head())
print()

# Đếm số sinh viên qua/rớt
print("So sinh vien theo ket qua:")
print(df["qua_mon"].value_counts())
print()

# Tỷ lệ qua môn
print("Ty le qua mon:", round(df["qua_mon"].mean(), 4))
print()

# Số giờ ôn trung bình của từng nhóm
print("So gio on trung binh theo nhom:")
print(df.groupby("qua_mon")["gio_on"].mean().round(2))