"""Bài 5: Chọn k hồi quy bằng CV 5 phần trên tập học."""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('data/gia_nha.csv')
X, y = df[['dien_tich']], df['gia']
X_hoc, X_kt, y_hoc, y_kt = train_test_split(X, y, test_size=0.2, random_state=42)
diem = {k: cross_val_score(KNeighborsRegressor(n_neighbors=k), X_hoc, y_hoc, cv=5, scoring='neg_mean_absolute_error').mean() for k in range(1, 21)}
k_tot = max(diem, key=diem.get)
knn = KNeighborsRegressor(n_neighbors=k_tot).fit(X_hoc, y_hoc)
linear = LinearRegression().fit(X_hoc, y_hoc)
mae_knn = mean_absolute_error(y_kt, knn.predict(X_kt))
mae_linear = mean_absolute_error(y_kt, linear.predict(X_kt))
print('BÀI 5 - KNN HỒI QUY')
print('MAE CV theo k (tỷ đồng):')
for k, score in diem.items():
    print(f'k={k:2d}, MAE CV={-score:.4f}')
print(f'k tốt nhất: {k_tot}; MAE CV={-diem[k_tot]:.4f}')
print(f'MAE KNN tập kiểm tra: {mae_knn:.4f}')
print(f'MAE đường thẳng tập kiểm tra: {mae_linear:.4f}')
print('KNN thắng đường thẳng?', mae_knn < mae_linear)
print('Nhận xét: MAE càng nhỏ càng tốt. KNN không ngoại suy tốt khi diện tích ngoài vùng đã học; hồi quy tuyến tính phù hợp hơn nếu quan hệ gần tuyến tính.')
