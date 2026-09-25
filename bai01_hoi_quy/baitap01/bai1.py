# -*- coding: utf-8 -*-
"""Bài 1: Lọc ra nhóm căn hộ lớn."""


import pandas as pd


df = pd.read_csv("data/gia_nha.csv")

nhom_lon = df[df["dien_tich"] > 100]

so_can = len(nhom_lon)

gia_tb = nhom_lon["gia"].mean()

print("So can dien tich lon hon 100 m2:", so_can)

print(f"Gia trung binh nhom nay: {gia_tb:.3f} ty dong")