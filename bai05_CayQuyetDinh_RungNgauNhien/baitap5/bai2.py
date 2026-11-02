# -*- coding: utf-8 -*-
"""Bài 2: tự duyệt tất cả cột và ngưỡng trên 450 khách của tập học."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def gini(nhan):
    nhan = np.asarray(nhan)
    if len(nhan) == 0:
        return 0.0
    p1 = nhan.mean()
    return 1 - p1 ** 2 - (1 - p1) ** 2


def tim_cau_hoi(X, y):
    """Trả về (tên cột, ngưỡng) có mức giảm Gini lớn nhất."""
    y = np.asarray(y)
    if len(X) != len(y) or len(y) < 2:
        raise ValueError("X, y phai cung so mau va co it nhat hai mau.")
    goc = gini(y)
    giam_tot = -np.inf
    cot_tot, nguong_tot = None, None
    for cot in X.columns:
        gia_tri = X[cot].to_numpy()
        muc = np.sort(np.unique(gia_tri))
        for a, b in zip(muc[:-1], muc[1:]):
            nguong = (float(a) + float(b)) / 2
            di_trai = gia_tri <= nguong
            n_trai = di_trai.sum()
            g_sau = (n_trai * gini(y[di_trai])
                     + (len(y) - n_trai) * gini(y[~di_trai])) / len(y)
            giam = goc - g_sau
            if giam > giam_tot:
                giam_tot, cot_tot, nguong_tot = giam, cot, nguong
    if cot_tot is None:
        raise ValueError("Khong co nguong hop le: tat ca cot deu la hang so.")
    return cot_tot, nguong_tot


def main():
    df = pd.read_csv("data/khach_vay.csv")
    X, y = df.drop(columns="tra_dung_han"), df["tra_dung_han"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    cot, nguong = tim_cau_hoi(X_tr, y_tr)
    nhan = y_tr.to_numpy()
    di_trai = X_tr[cot].to_numpy() <= nguong
    g_sau = (di_trai.sum() * gini(nhan[di_trai])
             + (~di_trai).sum() * gini(nhan[~di_trai])) / len(nhan)
    cay = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(X_tr, y_tr)
    cot_sklearn = X.columns[cay.tree_.feature[0]]
    nguong_sklearn = cay.tree_.threshold[0]
    print(f"So khach tap hoc: {len(X_tr)}")
    print(f"Tu tinh: {cot} <= {nguong:.2f}")
    print(f"Gini goc: {gini(nhan):.6f}")
    print(f"Gini sau chia: {g_sau:.6f}; Gini giam: {gini(nhan) - g_sau:.6f}")
    print(f"Nhanh trai: {di_trai.sum()} khach; nhanh phai: {(~di_trai).sum()} khach")
    print(f"Scikit-learn: {cot_sklearn} <= {nguong_sklearn:.2f}")
    assert cot == cot_sklearn and np.isclose(nguong, nguong_sklearn)
    assert cot == "thu_nhap" and np.isclose(nguong, 15.05)
    print("Ket qua trung nhau.")
    print("Giai thich: Chi can thu trung diem giua hai gia tri lien nhau,")
    print("vi moi nguong trong cung khoang chia du lieu giong nhau.")
    print("Phai tinh Gini hai nhanh co trong so theo so khach, khong lay")
    print("trung binh don gian. Gini goc co dinh nen giam Gini lon nhat")
    print("tuong duong Gini sau chia nho nhat. Chi dung tap hoc de tim nguong.")
    print("Ket luan nay ap dung cho cay Gini tren cung tap hoc; tung cay")
    print("trong rung co the co goc khac do boc mau va chon cot ngau nhien.")


if __name__ == "__main__":
    main()
