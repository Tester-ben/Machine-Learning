import pandas as pd


# Doc file du lieu
df = pd.read_csv("data/gia_nha.csv")

# 1. Kich thuoc bang
print("Kich thuoc bang (so dong, so cot):", df.shape)

print()

# 2. In 5 dong dau tien
print("Nam dong dau tien:")
print(df.head())

print()

# 3. Ten cac cot
print("Ten cac cot:")
print(list(df.columns))

print()

# 4. Thong ke nhanh dien_tich va gia
print("Thong ke nhanh cot dien_tich va cot gia:")
print(df[["dien_tich", "gia"]].describe().round(2))

print()

# 5. Kiem tra o bi thieu
print("So o bi thieu trong tung cot:")
print(df.isna().sum())