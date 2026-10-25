"""Bài 1: KNN tự viết chỉ với numpy (không dùng KNN scikit-learn)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    """Dự đoán nhãn bằng bỏ phiếu đa số trong k láng giềng Euclid."""
    X_hoc, y_hoc, x_moi = np.asarray(X_hoc), np.asarray(y_hoc), np.asarray(x_moi)
    khoang_cach = np.sqrt(np.sum((X_hoc - x_moi) ** 2, axis=1))
    chi_so = np.argsort(khoang_cach, kind='stable')[:k]
    nhan, so_phieu = np.unique(y_hoc[chi_so], return_counts=True)
    return nhan[np.argmax(so_phieu)]

df = pd.read_csv('data/hai_vang_trang.csv')
X, y = df[['x1', 'x2']], df['nhan']
X_hoc, X_kt, y_hoc, y_kt = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
k = 13
du_doan_tu_viet = np.array([knn_du_doan(X_hoc.to_numpy(), y_hoc.to_numpy(), x, k) for x in X_kt.to_numpy()])
mo_hinh = KNeighborsClassifier(n_neighbors=k).fit(X_hoc, y_hoc)
du_doan_thu_vien = mo_hinh.predict(X_kt)
print('BÀI 1 - TỰ VIẾT KNN')
print(f'Accuracy tự viết: {accuracy_score(y_kt, du_doan_tu_viet):.4f}')
print(f'Accuracy sklearn: {accuracy_score(y_kt, du_doan_thu_vien):.4f}')
print('Dự đoán trùng nhau:', np.array_equal(du_doan_tu_viet, du_doan_thu_vien))
print('Nhận xét: KNN tính khoảng cách từ mẫu mới tới từng mẫu học, lấy k điểm gần nhất rồi bỏ phiếu. Không cần fit mô hình phức tạp.')
