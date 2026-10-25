"""Bài 3: Đổi đơn vị chi tiêu và kiểm tra ảnh hưởng chuẩn hóa."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/sinh_vien_mo_rong.csv')
X, y = df[['gio_on', 'diem_giua_ky', 'chi_tieu']].copy(), df['qua_mon']
X_hoc, X_kt, y_hoc, y_kt = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)

def danh_gia(train, test, chuan_hoa):
    mh = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)) if chuan_hoa else KNeighborsClassifier(n_neighbors=7)
    mh.fit(train, y_hoc)
    return accuracy_score(y_kt, mh.predict(test))

X_hoc_moi, X_kt_moi = X_hoc.copy(), X_kt.copy()
X_hoc_moi['chi_tieu'] /= 1000
X_kt_moi['chi_tieu'] /= 1000
print('BÀI 3 - ĐỔI ĐƠN VỊ')
for chuan in [False, True]:
    cu = danh_gia(X_hoc, X_kt, chuan)
    moi = danh_gia(X_hoc_moi, X_kt_moi, chuan)
    print(f'Chuẩn hóa={chuan}: đơn vị nghìn={cu:.4f}, đơn vị triệu={moi:.4f}, chênh lệch={moi-cu:+.4f}')
print('Giải thích: Khi không chuẩn hóa, đổi đơn vị làm thay đổi đóng góp của chi_tieu vào khoảng cách Euclid. Khi chuẩn hóa, đổi đơn vị theo hệ số dương không làm thay đổi tọa độ z-score nên dự đoán giữ nguyên.')
