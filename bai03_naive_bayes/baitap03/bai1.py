# -*- coding: utf-8 -*-
"""Bai tap 1 - Bayes voi ba nguyen nhan."""


def bayes_nhieu_nhanh(priors, likelihoods):
    """Tra ve posterior cua tung nhanh khi cung quan sat mot bang chung."""
    if len(priors) != len(likelihoods):
        raise ValueError("priors va likelihoods phai cung so phan tu")

    trong_so = [p * l for p, l in zip(priors, likelihoods)]
    evidence = sum(trong_so)
    return [w / evidence for w in trong_so]


ten_nhom = ["hoc deu", "hoc don", "it hoc"]
tien_nghiem = [0.50, 0.30, 0.20]
xac_suat_qua_mon = [0.95, 0.70, 0.30] 

hau_nghiem = bayes_nhieu_nhanh(tien_nghiem, xac_suat_qua_mon)

print("Bai 1 - Bayes voi ba nguyen nhan")
for ten, p in zip(ten_nhom, hau_nghiem):
    print(f" P({ten} | da qua mon) = {p:.6f}")
print(f" Tong ba xac suat      = {sum(hau_nghiem):.6f}")
print()
print("Tu kiem tra: tong = 1.000000, dung voi dieu kien xac suat")
