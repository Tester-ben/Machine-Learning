"""Bài 6: Đo thời gian fit và predict; thời gian phụ thuộc máy chạy."""
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(0)
X_moi = rng.random((1000, 2))
print('BÀI 6 - THỜI GIAN KNN')
print('Số mẫu       fit (giây)       predict 1000 mẫu (giây)')
for n in [1000, 10000, 100000]:
    X_hoc = rng.random((n, 2))
    y_hoc = rng.integers(0, 2, size=n)
    # brute: minh họa trực tiếp việc KNN so khoảng cách với dữ liệu cũ
    mh = KNeighborsClassifier(n_neighbors=5, algorithm='brute', n_jobs=1)
    t0 = time.perf_counter()
    mh.fit(X_hoc, y_hoc)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    mh.predict(X_moi)
    t_pred = time.perf_counter() - t0
    print(f'{n:7d}       {t_fit:10.5f}       {t_pred:10.5f}')
print('Giải thích: KNN học lười (lazy learning): fit chủ yếu ghi nhớ dữ liệu, predict phải tìm láng giềng trên tập học nên tốn thời gian hơn khi số mẫu tăng. Thời gian thực tế tùy cấu hình máy.')
