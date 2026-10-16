# -*- coding: utf-8 -*-

import math
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv("data/tin_nhan.csv")

# Chia VAN BAN truoc. random_state=17 va stratify giup tai lap va giu ti le lop.
tin_train, tin_test, nhan_train, nhan_test = train_test_split(
    df["noi_dung"],
    df["la_rac"],
    test_size=0.25,
    random_state=17,
    stratify=df["la_rac"],
)

# ===== PHAN TU CAI DAT =====
# Từ điển chỉ được xây từ tập train.
tu_dien = sorted({tu for cau in tin_train for tu in str(cau).split()})
V = len(tu_dien)
tap_tu = set(tu_dien)

# Đếm số văn bản mỗi lớp -> prior.
dem_van_ban = Counter(int(v) for v in nhan_train)

# Đếm số lần từng từ xuất hiện trong mỗi lớp và tổng số token mỗi lớp.
dem_tu = {0: Counter(), 1: Counter()}
tong_tu = {0: 0, 1: 0}

for cau, nhan in zip(tin_train, nhan_train):
    nhan = int(nhan)
    for tu in str(cau).split():
        dem_tu[nhan][tu] += 1
        tong_tu[nhan] += 1


def log_diem(cau, nhan, alpha=1.0):
    """log P(nhan) + tong(count_tu * log P(tu | nhan))."""
    prior = dem_van_ban[nhan] / len(nhan_train)
    diem = math.log(prior)

    # Counter giúp từ lặp lại đóng góp đúng số lần, đúng mô hình Multinomial.
    for tu, so_lan in Counter(str(cau).split()).items():
        if tu not in tap_tu:
            # Từ hoàn toàn mới không nằm trong từ điển train -> bỏ qua.
            continue
        xac_suat_tu = (dem_tu[nhan][tu] + alpha) / (
            tong_tu[nhan] + alpha * V
        )
        diem += so_lan * math.log(xac_suat_tu)
    return diem


def du_doan(cau, alpha=1.0):
    diem_thuong = log_diem(cau, 0, alpha)
    diem_rac = log_diem(cau, 1, alpha)
    return 1 if diem_rac > diem_thuong else 0


du_doan_tu_viet = [du_doan(cau, alpha=1.0) for cau in tin_test]
acc_tu_viet = accuracy_score(nhan_test, du_doan_tu_viet)

# ===== DOI CHIEU VOI SCIKIT-LEARN =====
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X_train = vec.fit_transform(tin_train)
X_test = vec.transform(tin_test)  # Tuyet doi khong fit_transform tren test.

mo_hinh_sklearn = MultinomialNB(alpha=1.0)
mo_hinh_sklearn.fit(X_train, nhan_train)
du_doan_sklearn = mo_hinh_sklearn.predict(X_test)
acc_sklearn = accuracy_score(nhan_test, du_doan_sklearn)

so_khac = sum(
    int(a != b) for a, b in zip(du_doan_tu_viet, du_doan_sklearn)
)

print("Bai 4 - Tu viet bo loc khong dung MultinomialNB")
print(f"So tu trong tu dien train       = {V}")
print(f"Accuracy bo loc tu viet          = {acc_tu_viet:.4f}")
print(f"Accuracy sklearn MultinomialNB   = {acc_sklearn:.4f}")
print(f"So mau du doan khac sklearn      = {so_khac}")
print()
print()
print("Nhan xet:")

print(
    "- Bo loc tu viet cho ket qua rat gan "
    "MultinomialNB cua sklearn."
)

print(
    "- Laplace smoothing giup tranh xac suat bang 0."
)

print(
    "- Dung tong log giup tranh hien tuong underflow "
    "khi phai nhan rat nhieu xac suat nho."
)

print(
    "- Neu hai accuracy lech nhieu, can kiem tra "
    "cach tach tu, prior va cong thuc lam tron."
)
