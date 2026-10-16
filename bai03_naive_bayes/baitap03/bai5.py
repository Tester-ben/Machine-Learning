# -*- coding: utf-8 -*-
"""Bai tap 5 - Co tinh chon sai bien the Naive Bayes de thay loi im lang."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB


df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# MultinomialNB van chay vi du lieu o day khong am, nhung ve NGUYEN TAC la sai.
mo_hinh_sai = MultinomialNB()
mo_hinh_sai.fit(X_train, y_train)
acc_mnb = accuracy_score(y_test, mo_hinh_sai.predict(X_test))

mo_hinh_dung = GaussianNB()
mo_hinh_dung.fit(X_train, y_train)
acc_gnb = accuracy_score(y_test, mo_hinh_dung.predict(X_test))

print("Bai 5 - Chon sai bien the Naive Bayes")
print(f"MultinomialNB accuracy = {acc_mnb:.4f}")
print(f"GaussianNB    accuracy = {acc_gnb:.4f}")
print("Chuong trinh co bao loi? KHONG.")
print()
print("Giai thich:")
print(
    "- MultinomialNB duoc xay cho du lieu DEM khong am, vi du so lan mot tu xuat hien."
)
print(
    "- gio_on va diem_giua_ky la dai luong LIEN TUC, khong phai tan suat dem; "
    "y nghia xac suat cua MultinomialNB khong phu hop."
)
print(
    "- GaussianNB phu hop hon vi no mo ta moi dac trung lien tuc trong tung lop "
    "bang trung binh va phuong sai."
)
print(
    "- Day la loi im lang: chuong trinh van chay va van cho accuracy, "
    "nhung mo hinh sai ve gia dinh."
)
