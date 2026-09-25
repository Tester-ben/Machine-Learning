import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Đọc dữ liệu
df = pd.read_csv("data/gia_nha.csv")

# Hai biến đầu vào: diện tích + số phòng
X = df[["dien_tich", "so_phong"]]

# Biến cần dự đoán
y = df["gia"]

# Chia dữ liệu học và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính R2
r2 = r2_score(y_test, y_pred)

print("He so cua mo hinh:")
print("w1 (dien_tich) =", model.coef_[0])
print("w2 (so_phong) =", model.coef_[1])
print("b =", model.intercept_)

print()

print("R2 tren tap kiem tra =", round(r2, 4))

print()

print("So sanh:")
print("R2 mo hinh mot bien = 0.9622")
print("R2 mo hinh hai bien =", round(r2, 4))