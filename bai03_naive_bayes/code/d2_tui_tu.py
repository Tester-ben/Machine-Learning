# -*- coding: utf-8 -*-
"""Muc 3 - Doc kho tin nhan va bien van ban thanh bang so bang tui tu."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv("data/tin_nhan.csv")

print("Kich thuoc bang:", df.shape)
print()
print("So tin theo nhan (1 la rac, 0 la thuong):")
print(df["la_rac"].value_counts())
print()
print("Ty le tin rac:", round(df["la_rac"].mean(), 4))
print()

print("Ba tin nhan dau tien:")
for i in range(3):
    print(f" [{df.la_rac.iloc[i]}] {df.noi_dung.iloc[i]}")
print()


vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])

print("Kich thuoc bang dem tu:", X.shape)
print("So tu trong tu dien:", len(vec.vocabulary_))
print()

dem = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
dem["la_rac"] = df["la_rac"].to_numpy()
tong = dem.groupby("la_rac").sum().T
tong.columns = ["thuong", "rac"]

print("Muoi tu xuat hien nhieu nhat trong tin rac:")
print(tong.sort_values("rac", ascending=False).head(10))
print()
print("Muoi tu xuat hien nhieu nhat trong tin thuong:")
print(tong.sort_values("thuong", ascending=False).head(10))
