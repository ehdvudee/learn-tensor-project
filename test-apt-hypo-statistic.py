import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('./datas/노원_상계.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 1: -1]
y_data = xy[:, [-1]]

plt.scatter(x_data[:, [0]], y_data[:, [0]], alpha=0.1)
plt.xlabel("area")
plt.ylabel("sale price")
plt.savefig("./statistics/apt_area_price.png")
plt.show()

plt.scatter(x_data[:, [2]], y_data[:, [0]], alpha=0.1)
plt.xlabel("age of build")
plt.ylabel("sale price")
plt.savefig("./statistics/apt_ageOfBuild_price.png")
plt.show()


