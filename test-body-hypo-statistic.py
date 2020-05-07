import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xy = np.loadtxt('./datas/ret_body_data.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0: -1]
y_data = xy[:, [-1]]

plt.scatter(x_data[:, 0], y_data[:, 0], alpha=0.1)
plt.xlabel("height")
plt.ylabel("weight")
plt.savefig("./statistics/body_height_weight.png")
plt.show()

plt.scatter(x_data[:, 1], y_data[:, 0], alpha=0.1)
plt.xlabel("waist")
plt.ylabel("weight")
plt.savefig("./statistics/body_waist_weight.png")
plt.show()

