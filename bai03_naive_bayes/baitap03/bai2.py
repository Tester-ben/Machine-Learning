# -*- coding: utf-8 -*-
"""Bai tap 2 - Tim 5 tu to cao tin rac va 5 tu nghieng ve tin thuong manh nhat."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv("data/tin_nhan.csv")

vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

# Theo de bai: huan luyen tren TOAN BO 300 tin nhan.
mo_hinh = MultinomialNB(alpha=1.0)
mo_hinh.fit(X, y)

tu = vec.get_feature_names_out()
chenh_lech = mo_hinh.feature_log_prob_[1] - mo_hinh.feature_log_prob_[0]

chi_so_rac = np.argsort(chenh_lech)[-5:][::-1]
chi_so_thuong = np.argsort(chenh_lech)[:5]

print("Bai 2 - 5 tu to cao tin rac manh nhat")
for i in chi_so_rac:
    print(f" {tu[i]:12s}  log-ratio = {chenh_lech[i]: .4f}")

print()
print("5 tu cho thay tin thuong manh nhat")
for i in chi_so_thuong:
    print(f" {tu[i]:12s}  log-ratio = {chenh_lech[i]: .4f}")

print()
print("Giai thich: log-ratio > 0 nghieng ve rac, < 0 nghieng ve thuong.")
