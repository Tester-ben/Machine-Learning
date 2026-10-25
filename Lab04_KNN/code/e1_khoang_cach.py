# -*- coding: utf-8 -*-
"""LAB 04 KNN - Muc 2.3: Khoang cach Euclid.
Chay doc lap bang VS Code: python e1_khoang_cach.py
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

def buoc_1_khoang_cach():
    ba_ban = {
        'A': np.array([5.0, 3.0, 4000.0]),
        'B': np.array([25.0, 9.0, 4100.0]),
        'C': np.array([6.0, 3.5, 7000.0]),
    }

    def kc(u, v):
        return np.sqrt(((u - v) ** 2).sum())
    print('Khoang cach KHI CHUA chuan hoa')
    print(f" d(A, B) = {kc(ba_ban['A'], ba_ban['B']):8.1f}")
    print(f" d(A, C) = {kc(ba_ban['A'], ba_ban['C']):8.1f}")
    print()

    df = pd.read_csv(DATA_DIR / 'sinh_vien_mo_rong.csv')
    X = df[['gio_on', 'diem_giua_ky', 'chi_tieu']].to_numpy(dtype=float)
    tb, dl = X.mean(axis=0), X.std(axis=0)
    print('Do lech chuan cua ba cot:')
    print(f' gio_on {dl[0]:.2f}   diem_giua_ky {dl[1]:.2f}   chi_tieu {dl[2]:.0f}')
    print()
    chuan = {ten: (v - tb) / dl for ten, v in ba_ban.items()}
    print('Khoang cach SAU KHI chuan hoa')
    print(f" d(A, B) = {kc(chuan['A'], chuan['B']):.2f}")
    print(f" d(A, C) = {kc(chuan['A'], chuan['C']):.2f}")
    print()

    X2 = df[['gio_on', 'diem_giua_ky']].to_numpy(dtype=float)
    y = df['qua_mon'].to_numpy()
    tb2, dl2 = X2.mean(axis=0), X2.std(axis=0)
    ban_moi = np.array([12.0, 6.5])
    X2c = (X2 - tb2) / dl2
    moi_c = (ban_moi - tb2) / dl2
    khoang = np.sqrt(((X2c - moi_c) ** 2).sum(axis=1))
    gan = np.argsort(khoang)[:5]
    print('Ban moi: on 12 gio, diem giua ky 6.5')
    print('Nam lang gieng gan nhat:')
    for i in gan:
        ket = 'qua' if y[i] == 1 else 'rot'
        print(f' gio {X2[i, 0]:5.1f} diem {X2[i, 1]:4.1f} '
              f'khoang cach {khoang[i]:.3f} -> {ket}')
    so_qua = int(y[gan].sum())
    print(f'Bo phieu: {so_qua} qua, {5 - so_qua} rot')
    print('Du doan:', 'qua mon' if so_qua >= 3 else 'rot mon')


if __name__ == "__main__":
    buoc_1_khoang_cach()
