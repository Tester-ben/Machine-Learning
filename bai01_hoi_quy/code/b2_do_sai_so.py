import pandas as pd


# Doc du lieu
df = pd.read_csv("data/gia_nha.csv")


# Lay 5 can nha dai dien
nho = df.iloc[[0, 14, 29, 44, 59]]


x = nho["dien_tich"].to_numpy()
y = nho["gia"].to_numpy()



# Ham tinh MSE
def mse(w, b):

    # Gia du doan
    y_du_doan = w * x + b

    # Sai so
    sai_so = y - y_du_doan

    # MSE
    return (sai_so ** 2).mean()



# In 5 can nha
print("Nam can duoc chon:")

for i in range(5):
    print(
        f"{x[i]:6.1f} m2 -> {y[i]:5.2f} ty"
    )


print()


# Thu 2 duong thang
for w, b in [(0.05, 1.5),(0.08, 0.5)]:

    print(
        f"Duong thang y = {w} * x + {b}"
    )

    for i in range(5):

        du_doan = w*x[i] + b

        sai_so = y[i] - du_doan

        print(
            f"x={x[i]:6.1f} "
            f"that={y[i]:5.2f} "
            f"du_doan={du_doan:5.2f} "
            f"sai_so={sai_so:+6.2f}"
        )

    print(f"MSE = {mse(w,b):.4f}")
    print()

print("Duong nao co MSE nho hon thi duong do khop du lieu tot hon.")