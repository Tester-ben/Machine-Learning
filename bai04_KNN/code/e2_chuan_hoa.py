# -*- coding: utf-8 -*-
"""LAB 04 KNN - Muc 3: Chuan hoa.
Chay doc lap bang VS Code: python e2_chuan_hoa.py
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

def buoc_2_chuan_hoa():
    """Muc 3: Do hau qua cua viec quen chuan hoa."""
    df = pd.read_csv(DATA_DIR / 'sinh_vien_mo_rong.csv')
    y = df['qua_mon']
    X3 = df[['gio_on', 'diem_giua_ky', 'chi_tieu']]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X3, y, test_size=0.25, random_state=17, stratify=y
    )

    def cham(mo_hinh, cot):
        mo_hinh.fit(X_tr[cot], y_tr)
        return accuracy_score(y_te, mo_hinh.predict(X_te[cot]))

    def knn_chuan():
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))

    ba_cot = ['gio_on', 'diem_giua_ky', 'chi_tieu']
    hai_cot = ['gio_on', 'diem_giua_ky']
    a = cham(KNeighborsClassifier(n_neighbors=7), ba_cot)
    b = cham(knn_chuan(), ba_cot)
    c = cham(knn_chuan(), hai_cot)
    print(f'Doan bua ca lop qua          : {y_te.mean():.4f}')
    print(f'Ba cot, khong chuan hoa      : {a:.4f}')
    print(f'Ba cot, co chuan hoa         : {b:.4f}')
    print(f'Bo cot chi_tieu, co chuan hoa: {c:.4f}')


if __name__ == "__main__":
    buoc_2_chuan_hoa()
