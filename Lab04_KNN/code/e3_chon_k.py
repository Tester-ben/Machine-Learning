# -*- coding: utf-8 -*-
"""LAB 04 KNN - Muc 4: Chon k.
Chay doc lap bang VS Code: python e3_chon_k.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def tieu_de(so, ten):
    print("\n" + "=" * 70)
    print(f"BUOC {so}: {ten}")
    print("=" * 70)

def buoc_3_chon_k():
    """Muc 4: Chon k bang kiem dinh cheo 5 phan."""
    df = pd.read_csv(DATA_DIR / 'hai_vang_trang.csv')
    X, y = df[['x1', 'x2']], df['nhan']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    print('Accuracy tren tap hoc va tap kiem tra theo k')
    print('   k    tap hoc   tap kiem tra')
    for k in [1, 5, 13, 31, 101]:
        mh = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
        a_hoc = accuracy_score(y_tr, mh.predict(X_tr))
        a_kt = accuracy_score(y_te, mh.predict(X_te))
        print(f' {k:3d}    {a_hoc:.4f}      {a_kt:.4f}')
    print('\nKiem dinh cheo 5 phan tren tap hoc:')
    diem = {}
    for k in range(1, 52, 2):
        diem[k] = cross_val_score(
            KNeighborsClassifier(n_neighbors=k), X_tr, y_tr, cv=5
        ).mean()
    k_tot = max(diem, key=diem.get)
    for k in [1, 5, 9, 13, 17, 25, 51]:
        dau = ' <- tot nhat' if k == k_tot else ''
        print(f' k = {k:2d} diem trung binh {diem[k]:.4f}{dau}')
    mh = KNeighborsClassifier(n_neighbors=k_tot).fit(X_tr, y_tr)
    print(f'\nk tot nhat = {k_tot}')
    print(f'Accuracy tren tap kiem tra = {accuracy_score(y_te, mh.predict(X_te)):.4f}')


if __name__ == "__main__":
    buoc_3_chon_k()
