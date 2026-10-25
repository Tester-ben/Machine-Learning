# -*- coding: utf-8 -*-
"""LAB 04 KNN - Muc 6: Hoi quy.
Chay doc lap bang VS Code: python e5_knn_hoi_quy.py
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

def buoc_5_hoi_quy():
    """Muc 6: KNN hoi quy gia nha va gioi han ngoai suy."""
    df = pd.read_csv(DATA_DIR / 'gia_nha.csv')
    X, y = df[['dien_tich']], df['gia']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    knn = KNeighborsRegressor(n_neighbors=5).fit(X_tr, y_tr)
    duong = LinearRegression().fit(X_tr, y_tr)
    mae_knn = mean_absolute_error(y_te, knn.predict(X_te))
    mae_duong = mean_absolute_error(y_te, duong.predict(X_te))
    print('MAE tren tap kiem tra (ty dong)')
    print(f' KNN, k = 5 : {mae_knn:.4f}')
    print(f' Duong thang : {mae_duong:.4f}')
    can_moi = pd.DataFrame({'dien_tich': [60.0, 100.0, 150.0, 250.0]})
    print('\nDu doan cho bon can moi (ty dong)')
    print(' dien tich      KNN   duong thang')
    for dt, a, b in zip(can_moi['dien_tich'], knn.predict(can_moi), duong.predict(can_moi)):
        print(f' {dt:6.0f} m2    {a:6.2f}   {b:6.2f}')
    _, chi_so = knn.kneighbors(pd.DataFrame({'dien_tich': [250.0]}))
    lg = X_tr.iloc[chi_so[0]]['dien_tich'].tolist()
    print('\nNam lang gieng cua can 250 m2:', ', '.join(f'{v:.1f}' for v in lg))


if __name__ == "__main__":
    buoc_5_hoi_quy()
