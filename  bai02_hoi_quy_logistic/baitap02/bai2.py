# -*- coding: utf-8 -*-
"""Bai tap 2: Ve ham sigmoid."""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-8, 8, 200)
y = sigmoid(z)

plt.plot(z, y, label="Sigmoid")


plt.axhline(y=0.5, linestyle="--", label="y = 0.5")

plt.axvline(x=0, linestyle="--", label="z = 0")

plt.title("Ham sigmoid")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")


plt.xlim(-8, 8)
plt.ylim(0, 1)

plt.legend()
plt.grid()

plt.savefig("sigmoid.png")

plt.show()