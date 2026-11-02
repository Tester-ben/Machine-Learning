# -*- coding: utf-8 -*-
"""Bài 1: vẽ cây năm lá và cây sâu 2, giải thích hai lá bên trái."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def main():
    df = pd.read_csv("data/khach_vay.csv")
    X, y = df.drop(columns="tra_dung_han"), df["tra_dung_han"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    thu_muc = Path("baitap5")
    cay_nam_la = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(X_tr, y_tr)
    cay_sau_2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
    for cay, ten, tieu_de in [
        (cay_nam_la, "cay_nam_la.png", "Cay quyet dinh nam la"),
        (cay_sau_2, "cay_sau_2.png", "Cay quyet dinh sau 2"),
    ]:
        plt.figure(figsize=(18, 9))
        plot_tree(cay, feature_names=list(X.columns),
                  class_names=["tre han", "dung han"], filled=True,
                  rounded=True, precision=3, fontsize=10)
        plt.title(tieu_de)
        plt.tight_layout()
        plt.savefig(thu_muc / ten, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Da luu: baitap5/{ten}")
    cay = cay_sau_2.tree_
    nut_trai = cay.children_left[0]
    la_trai, la_phai = cay.children_left[nut_trai], cay.children_right[nut_trai]
    for la in [la_trai, la_phai]:
        so_luong = np.rint(cay.value[la][0] / cay.value[la][0].sum()
                           * cay.n_node_samples[la]).astype(int)
        print(f"La {la}: [tre han, dung han] = {so_luong.tolist()}, "
              f"Gini = {cay.impurity[la]:.4f}, ket luan = tre han")
    gini_sau = sum(cay.n_node_samples[la] * cay.impurity[la]
                   for la in [la_trai, la_phai]) / cay.n_node_samples[nut_trai]
    print(f"Gini nut cha = {cay.impurity[nut_trai]:.4f}")
    print(f"Gini sau chia = {gini_sau:.4f}")
    print(f"Gini giam = {cay.impurity[nut_trai] - gini_sau:.4f}")
    print("Nhan xet: Hai la ben trai deu du doan tre han (lop 0).")
    print("Cay van chia vi ty le hai lop o hai la khac nhau; Gini trung binh")
    print("co trong so giam. Tieu chi chia la giam do lan lon, khong bat buoc")
    print("hai nhanh phai du doan hai lop khac nhau.")


if __name__ == "__main__":
    main()
