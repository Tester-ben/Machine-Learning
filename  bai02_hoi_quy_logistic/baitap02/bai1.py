# -*- coding: utf-8 -*-

import pandas as pd

# Doc du lieu
df = pd.read_csv("data/sinh_vien.csv")

# Chia thanh hai nhom
nhom_7_tro_len = df[df["diem_giua_ky"] >= 7]
nhom_con_lai = df[df["diem_giua_ky"] < 7]

# 1. So ban co diem giua ky tu 7 tro len
so_ban_7_tro_len = len(nhom_7_tro_len)

# 2. Ty le qua mon cua nhom diem giua ky >= 7
ty_le_qua_nhom_7 = nhom_7_tro_len["qua_mon"].mean()

# 3. Ty le qua mon cua nhom con lai
ty_le_qua_nhom_con_lai = nhom_con_lai["qua_mon"].mean()

# In ket qua
print("KET QUA BAI TAP 1")
print("-----------------------------")
print("So ban co diem giua ky tu 7 tro len:", so_ban_7_tro_len)
print(f"Ty le qua mon cua nhom >= 7: {ty_le_qua_nhom_7:.4f}")
print(f"Ty le qua mon cua nhom < 7 : {ty_le_qua_nhom_con_lai:.4f}")
print()

# Nhan xet
if ty_le_qua_nhom_7 > ty_le_qua_nhom_con_lai:
    print("Nhan xet: Nhom co diem giua ky tu 7 tro len co ty le qua mon cao hon.")
    print("Diem giua ky co kha nang phan biet hai nhom.")
elif ty_le_qua_nhom_7 < ty_le_qua_nhom_con_lai:
    print("Nhan xet: Nhom co diem giua ky tu 7 tro len co ty le qua mon thap hon.")
else:
    print("Nhan xet: Hai nhom co ty le qua mon bang nhau.")
    print("Diem giua ky chua phan biet ro hai nhom.")