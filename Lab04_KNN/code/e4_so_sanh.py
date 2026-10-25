# -*- coding: utf-8 -*-
"""LAB 04 KNN - Muc 5: So sanh mo hinh.
Chay doc lap bang VS Code: python e4_so_sanh.py
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

def buoc_4_so_sanh():
    """Muc 5: So sanh KNN voi Logistic Regression va Naive Bayes."""
    df = pd.read_csv(DATA_DIR / 'hai_vang_trang.csv')
    X, y = df[['x1', 'x2']], df['nhan']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    cac_mo_hinh = [
        ('Hoi quy logistic', LogisticRegression()),
        ('Naive Bayes     ', GaussianNB()),
        ('KNN, k = 13     ', KNeighborsClassifier(n_neighbors=13)),
    ]
    print('Accuracy tren tap kiem tra, bo hai vang trang')
    for ten, mh in cac_mo_hinh:
        mh.fit(X_tr, y_tr)
        print(f' {ten}: {accuracy_score(y_te, mh.predict(X_te)):.4f}')


if __name__ == "__main__":
    buoc_4_so_sanh()
