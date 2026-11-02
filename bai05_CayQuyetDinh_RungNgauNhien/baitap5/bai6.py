# -*- coding: utf-8 -*-
"""Bài 6: so rừng hồi quy 200 cây, cây sâu 3 và hồi quy tuyến tính."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def main():
    df = pd.read_csv("data/gia_nha.csv")
    X, y = df[["dien_tich"]], df["gia"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    mo_hinh = {
        "Cay sau 3": DecisionTreeRegressor(max_depth=3, random_state=0),
        "Duong thang": LinearRegression(),
        "Rung 200 cay": RandomForestRegressor(n_estimators=200, random_state=0),
    }
    moi = pd.DataFrame({"dien_tich": [250.0, 350.0]})
    print(f"Dien tich tap hoc: {X_tr.dien_tich.min():.1f} - {X_tr.dien_tich.max():.1f} m2")
    print("Mo hinh         MAE (ty dong)   Gia 250 m2   Gia 350 m2")
    sai_so = {}
    for ten, mh in mo_hinh.items():
        mh.fit(X_tr, y_tr)
        mae = mean_absolute_error(y_te, mh.predict(X_te))
        sai_so[ten] = mae
        du_doan = mh.predict(moi)
        print(f"{ten:14s}  {mae:.4f}          {du_doan[0]:.4f}       {du_doan[1]:.4f}")
    rung = mo_hinh["Rung 200 cay"]
    gia_tung_cay = np.array([cay.predict(moi.to_numpy())[0] for cay in rung.estimators_])
    print(f"\nTrung binh du doan 250 m2 cua 200 cay: {gia_tung_cay.mean():.4f} ty")
    print(f"Mo hinh co MAE thap nhat: {min(sai_so, key=sai_so.get)}")
    print("Nhan xet: Rung khong ngoai suy xu huong tang gia theo dien tich.")
    print("Moi cay du doan bang trung binh gia cac mau hoc roi vao la (co trong so")
    print("do boc mau bootstrap). Nha 250 va 350 m2 deu vuot mien dien tich tap hoc,")
    print("nen di vao cung la ngoai cung cua tung cay. Rung lay trung binh 200")
    print("du doan do, vi vay gia khong tiep tuc tang nhu duong thang.")


if __name__ == "__main__":
    main()
