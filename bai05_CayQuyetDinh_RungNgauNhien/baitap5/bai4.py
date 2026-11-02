# -*- coding: utf-8 -*-
"""Bài 4: chọn min_samples_leaf bằng kiểm định chéo 5 phần."""
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    df = pd.read_csv("data/khach_vay.csv")
    X, y = df.drop(columns="tra_dung_han"), df["tra_dung_han"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    diem = {}
    print("min_samples_leaf   accuracy CV 5 phan")
    for so_khach in [1, 2, 5, 10, 20, 40]:
        mh = DecisionTreeClassifier(min_samples_leaf=so_khach, random_state=0)
        diem[so_khach] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
        print(f"{so_khach:16d}   {diem[so_khach]:.4f}")
    tot = max(diem, key=diem.get)
    cay_la = DecisionTreeClassifier(min_samples_leaf=tot, random_state=0).fit(X_tr, y_tr)
    cay_sau = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_tr, y_tr)
    # Chỉ chấm mô hình đã chọn, không chấm từng giá trị thử trên tập kiểm tra.
    acc_la = cay_la.score(X_te, y_te)
    acc_sau = cay_sau.score(X_te, y_te)
    print(f"\nGia tri tot nhat: min_samples_leaf = {tot}, CV = {diem[tot]:.4f}")
    print(f"Cay duoc chon: do sau {cay_la.get_depth()}, {cay_la.get_n_leaves()} la")
    print(f"Accuracy kiem tra, cat tia theo so khach: {acc_la:.4f}")
    print(f"Accuracy kiem tra, cay sau 5: {acc_sau:.4f}")
    if acc_la > acc_sau:
        print("Nhan xet: Cat tia bang min_samples_leaf cho ket qua tot hon tren tap nay.")
    elif acc_la < acc_sau:
        print("Nhan xet: Gioi han max_depth=5 cho ket qua tot hon tren tap nay.")
    else:
        print("Nhan xet: Hai cach cho accuracy bang nhau tren tap kiem tra nay.")
    print("min_samples_leaf ngan la qua it mau; max_depth gioi han so tang cau hoi.")


if __name__ == "__main__":
    main()
