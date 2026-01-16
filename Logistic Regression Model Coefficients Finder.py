import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("init_points.csv")  # or task1_points.csv

X = df[["x", "y"]].values
y = df["class"].values

# Multiclass logistic regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        random_state=0
    ))
])
model.fit(X, y)

lr = model.named_steps["lr"]
scaler = model.named_steps["scaler"]

# Convert coefficients back to original scale
w_scaled = lr.coef_
b_scaled = lr.intercept_
w = w_scaled / scaler.scale_
b = b_scaled - np.sum(w_scaled * (scaler.mean_ / scaler.scale_), axis=1)

print("=== Logistic Regression Coefficients ===")
for cls, wi, bi in zip(lr.classes_, w, b):
    print(f"Class {cls}: w_x={wi[0]:.6f}, w_y={wi[1]:.6f}, intercept={bi:.6f}")

# Visualization
x_min, x_max = df["x"].min() - 1, df["x"].max() + 1
y_min, y_max = df["y"].min() - 1, df["y"].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, Z, alpha=0.25)

for cls in sorted(df["class"].unique()):
    sub = df[df["class"] == cls]
    plt.scatter(sub["x"], sub["y"], label=f"Class {cls}", edgecolors="k")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Multiclass Logistic Regression – Decision Regions")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
