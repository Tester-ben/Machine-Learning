# -*- coding: utf-8 -*-
"""Bài 2: Vẽ biểu đồ phân tán theo số phòng."""

import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu

df = pd.read_csv("data/gia_nha.csv")

# Vẽ biểu đồ phân tán

plt.scatter(
    df["so_phong"],
    df["gia"]
)


plt.xlabel("Số phòng ngủ")
plt.ylabel("Giá căn hộ (tỷ đồng)")
plt.title("Mối quan hệ giữa số phòng và giá căn hộ")

# Lưu hình

plt.savefig(
    "bai2.png",
    dpi=150
)

# Nhận xét

print(
    "Nhận xét: Biểu đồ cho thấy số phòng ngủ có xu hướng tăng cùng với giá căn hộ. "
    "Những căn có nhiều phòng thường có giá cao hơn, tuy nhiên vẫn có sự dao động do còn phụ thuộc vào các yếu tố khác như diện tích và tuổi nhà."
)

# Hiển thị
plt.show()