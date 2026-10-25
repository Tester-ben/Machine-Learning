"""Bài 4: Bỏ phiếu đều và bỏ phiếu theo nghịch đảo khoảng cách."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/hai_vang_trang.csv')
X, y = df[['x1', 'x2']], df['nhan']
X_hoc, X_kt, y_hoc, y_kt = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
print('BÀI 4 - LÁNG GIỀNG CÓ TRỌNG SỐ')
print('  k    uniform    distance')
for k in [1, 13, 101]:
    diem = []
    for weights in ['uniform', 'distance']:
        mh = KNeighborsClassifier(n_neighbors=k, weights=weights).fit(X_hoc, y_hoc)
        diem.append(accuracy_score(y_kt, mh.predict(X_kt)))
    print(f'{k:3d}    {diem[0]:.4f}     {diem[1]:.4f}')
print('Giải thích: Với k=101, weights=distance giảm ảnh hưởng của điểm xa; những láng giềng gần có phiếu nặng hơn, nên thường giữ được hình dạng biên cục bộ tốt hơn.')
