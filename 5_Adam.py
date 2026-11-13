import numpy as np
import matplotlib.pyplot as plt

def loss_function(x, y):
    return 0.1 * x**2 + 2.0 * y**2

def gradient(x, y):
    return np.array([0.2 * x, 4.0 * y])

start_pos = np.array([-8.0, 2.5])
steps = 40
lr = 0.4
beta1 = 0.9
beta2 = 0.999

pos = start_pos.copy()
m = np.zeros(2)
v = np.zeros(2)
path = [pos.copy()]
for t in range(1, steps + 1):
    grad = gradient(pos, pos)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    pos -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    path.append(pos.copy())
path = np.array(path)

x_grid = np.linspace(-9, 9, 100)
y_grid = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = loss_function(X, Y)

plt.figure(figsize=(8, 5))
plt.contour(X, Y, Z, levels=25, cmap='gray', alpha=0.4)
plt.plot(path[:, 0], path[:, 1], marker='o', markersize=4, color='orange', linewidth=2, label='Path: Adam')
plt.scatter(start_pos, start_pos, color='black', s=80, zorder=5, label='Start')
plt.scatter(0, 0, color='gold', edgecolor='black', marker='*', s=200, zorder=5, label='Minimum')

plt.title('5. Adam Optimizer', fontsize=13, fontweight='bold')
plt.xlabel('W1')
plt.ylabel('W2')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
