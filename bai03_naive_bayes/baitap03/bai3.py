# -*- coding: utf-8 -*-
"""Bai tap 3 - Doi alpha va theo doi accuracy, precision, recall."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv("data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

# Bám đúng tài liệu để đối chiếu kết quả.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

alpha_values = [0.001, 0.1, 1.0, 10.0, 100.0]

print("Bai 3 - Anh huong cua alpha")
print(f"{'alpha':>8s} {'accuracy':>10s} {'precision':>10s} {'recall':>10s}")
for alpha in alpha_values:
    mo_hinh = MultinomialNB(alpha=alpha)
    mo_hinh.fit(X_train, y_train)
    y_pred = mo_hinh.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1)

    print(f"{alpha:8.3f} {accuracy:10.4f} {precision:10.4f} {recall:10.4f}")

print()
print("Nhan xet:")
print(
    "Alpha qua lon lam cac xac suat tu o hai lop bi keo lai gan nhau, "
    "nen dau hieu phan loai bi lam phang."
)
print(
    "Tren bo test nay, alpha tu 0.001 den 100 van cho cung nhan du doan, "
    "nhung ve nguyen tac alpha qua lon co the lam giam kha nang phan biet."
)
