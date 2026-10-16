# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv("data/tin_nhan.csv")

tin_thuong = df[df["la_rac"] == 0]
tin_rac_20 = df[df["la_rac"] == 1].head(20)
df_lech = pd.concat([tin_thuong, tin_rac_20], ignore_index=True)

print("Bai 6 - Du lieu mat can bang")
print("So mau theo lop:")
print(df_lech["la_rac"].value_counts().sort_index())
print()

# Chia raw text trước để vectorizer chỉ học từ tập train.
tin_train, tin_test, y_train, y_test = train_test_split(
    df_lech["noi_dung"],
    df_lech["la_rac"],
    test_size=0.25,
    random_state=17,
    stratify=df_lech["la_rac"],
)

vec = CountVectorizer(token_pattern=r"(?u)\S+")
X_train = vec.fit_transform(tin_train)
X_test = vec.transform(tin_test)

mo_hinh = MultinomialNB(alpha=1.0)
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall_rac = recall_score(y_test, y_pred, pos_label=1)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"Accuracy         = {accuracy:.4f}")
print(f"Recall lop rac   = {recall_rac:.4f}")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print()
print("Giai thich:")
print(
    "1) Bo moi co 180 tin thuong va chi 20 tin rac, nen lop rac chi chiem 10%. "
    "Prior cua lop thuong lon hon rat nhieu."
)
print(
    "2) Accuracy = 0.9800 rat cao nhung recall rac = 0.8000: "
    "mo hinh van de lot 1/5 tin rac trong tap test."
)
print(
    "3) Neu mot mo hinh doan tat ca la thuong, accuracy da co the dat 90% "
    "nhung recall rac = 0. Do do accuracy mot minh khong phan anh dung chat luong bo loc."
)
print(
    "4) Voi bo loc thu rac, recall cua lop rac va confusion matrix can duoc xem cung accuracy, "
    "dac biet khi lop rac hiem."
)
