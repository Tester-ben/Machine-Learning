# -*- coding: utf-8 -*-
"""Bài 5: so độ quan trọng Gini và permutation trên tập kiểm tra."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("data/khach_vay.csv")
    X, y = df.drop(columns="tra_dung_han"), df["tra_dung_han"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    # Giữ nguyên chuỗi số của tập học như f6, rồi rút tiếp cho tập kiểm tra.
    rng = np.random.default_rng(0)
    X_tr_nn, X_te_nn = X_tr.copy(), X_te.copy()
    X_tr_nn["so_ngau_nhien"] = rng.random(len(X_tr))
    X_te_nn["so_ngau_nhien"] = rng.random(len(X_te))
    rung_nn = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr_nn, y_tr)
    mdi = pd.Series(rung_nn.feature_importances_, index=X_tr_nn.columns)
    ket_qua = permutation_importance(
        rung_nn, X_te_nn, y_te, n_repeats=20, random_state=0, scoring="accuracy"
    )
    pi = pd.Series(ket_qua.importances_mean, index=X_te_nn.columns)
    std = pd.Series(ket_qua.importances_std, index=X_te_nn.columns)
    bang = pd.DataFrame({
        "Gini": mdi, "Hang_Gini": mdi.rank(ascending=False, method="min").astype(int),
        "Permutation_mean": pi, "Permutation_std": std,
        "Hang_permutation": pi.rank(ascending=False, method="min").astype(int),
    }).sort_values("Permutation_mean", ascending=False)
    print(f"Accuracy kiem tra truoc xao tron: {rung_nn.score(X_te_nn, y_te):.4f}")
    print(bang.round(4).to_string())
    cot = "so_ngau_nhien"
    print(f"\nCot ngau nhien: hang Gini = {bang.loc[cot, 'Hang_Gini']}, "
          f"hang permutation = {bang.loc[cot, 'Hang_permutation']}")
    print("Giai thich: Gini cong muc giam do lan lon tren tap hoc. Cot ngau nhien")
    print("co nhieu gia tri, nhieu nguong thu nen de tach duoc vai mau nhieu va")
    print("nhan diem quan trong cao du khong co thong tin du bao that.")
    print("Permutation do muc giam accuracy khi xao tron tung cot tren tap kiem tra.")
    print("Cot ngau nhien co diem gan 0 (co the am), nen it giup du doan mau moi.")
    print("Diem am nghia la xao tron lai tang accuracy mot chut; do nhieu va mau huu han,")
    print("khong phai bang chung nhan qua. Thu hang nay chi thuoc lan chia du lieu nay.")


if __name__ == "__main__":
    main()
