"""Bài 2: So sánh Manhattan và Euclid qua kiểm định chéo."""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/hai_vang_trang.csv')
X, y = df[['x1', 'x2']], df['nhan']
X_hoc, X_kt, y_hoc, y_kt = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
print('BÀI 2 - MANHATTAN SO VỚI EUCLID')
for metric in ['euclidean', 'manhattan']:
    diem_cv = {k: cross_val_score(KNeighborsClassifier(n_neighbors=k, metric=metric), X_hoc, y_hoc, cv=5).mean() for k in range(1, 52, 2)}
    k_tot = max(diem_cv, key=diem_cv.get)
    mo_hinh = KNeighborsClassifier(n_neighbors=k_tot, metric=metric).fit(X_hoc, y_hoc)
    print(f'{metric:10s}: k tốt nhất={k_tot:2d}, CV={diem_cv[k_tot]:.4f}, accuracy kiểm tra={accuracy_score(y_kt,mo_hinh.predict(X_kt)):.4f}')
print('Nhận xét: Đối chiếu accuracy để xem đổi cách đo khoảng cách ảnh hưởng đáng kể hay không; k được chọn hoàn toàn từ tập học.')
