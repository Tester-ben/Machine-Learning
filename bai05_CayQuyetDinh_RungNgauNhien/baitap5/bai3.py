# -*- coding: utf-8 -*-
"""Bài 3: chọn độ sâu bằng CV 5 phần, so Gini và entropy."""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    df = pd.read_csv("data/khach_vay.csv")
    X, y = df.drop(columns="tra_dung_han"), df["tra_dung_han"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    ket_qua = {}
    for tieu_chi in ["gini", "entropy"]:
        print(f"\nTieu chi: {tieu_chi}")
        tu_do = DecisionTreeClassifier(criterion=tieu_chi, random_state=0).fit(X_tr, y_tr)
        print(f"Cay tu do: sau {tu_do.get_depth()}, {tu_do.get_n_leaves()} la")
        print(f"Accuracy hoc: {tu_do.score(X_tr, y_tr):.4f}; "
              f"kiem tra: {tu_do.score(X_te, y_te):.4f}")
        print("Do sau   tap hoc   kiem dinh cheo")
        diem = {}
        for d in range(1, 16):
            mh = DecisionTreeClassifier(criterion=tieu_chi, max_depth=d, random_state=0)
            diem[d] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
            hoc = mh.fit(X_tr, y_tr).score(X_tr, y_tr)
            print(f"{d:6d}   {hoc:.4f}    {diem[d]:.4f}")
        d_tot = max(diem, key=diem.get)
        tot = DecisionTreeClassifier(criterion=tieu_chi, max_depth=d_tot, random_state=0)
        tot.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, tot.predict(X_te))
        ket_qua[tieu_chi] = (d_tot, diem[d_tot], acc)
    print("\nTong ket: tieu chi / do sau tot nhat / CV / accuracy kiem tra")
    for ten, (d, cv, acc) in ket_qua.items():
        print(f"{ten:8s} {d:3d} {cv:.4f} {acc:.4f}")
    chenh = (ket_qua["entropy"][2] - ket_qua["gini"][2]) * 100
    print(f"Nhan xet: Entropy chenh Gini {chenh:+.2f} diem phan tram tren tap")
    print("kiem tra; muc chenh nho tren lan chia nay, khong phai cai thien lon.")
    print("Do sau duoc chon chi tu CV tap hoc; khong chon theo diem kiem tra.")


if __name__ == "__main__":
    main()
