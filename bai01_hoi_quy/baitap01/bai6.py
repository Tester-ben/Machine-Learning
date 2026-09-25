# -*- coding: utf-8 -*-
"""Bài 6: Viết hàm dự đoán có cảnh báo."""


# Hai hệ số đã tìm được từ mô hình một biến

w = 0.078367

b = 0.401752


def du_doan_gia(dien_tich):

    # Cảnh báo nếu diện tích nằm ngoài phạm vi dữ liệu

    if dien_tich < 35.5 or dien_tich > 117.5:

        print(
            f"Canh bao: {dien_tich} m2 nam ngoai khoang du lieu 35.5 - 117.5 m2."
        )

    # Tính giá dự đoán

    gia_du_doan = w * dien_tich + b

    return gia_du_doan


# Thử với 60, 80 và 200 m2

gia_60 = du_doan_gia(60)

print(f"60 m2 -> gia du doan = {gia_60:.3f} ty dong")

print()

gia_80 = du_doan_gia(80)

print(f"80 m2 -> gia du doan = {gia_80:.3f} ty dong")

print()


gia_200 = du_doan_gia(200)

print(f"200 m2 -> gia du doan = {gia_200:.3f} ty dong")