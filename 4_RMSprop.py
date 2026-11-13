import numpy as np
import matplotlib.pyplot as plt

def loss_function(x, y):
    return 0.1 * x**2 + 2.0 * y**2

def gradient(x, y):
    return np.array([0.2 * x, 4.0 * y])

start_pos = np.array([-8.0, 2.5])
steps = 40
lr = 0.3
beta = 0.9

pos = start_pos.copy()
sq_grad = np.zeros(2)
path = [pos.copy()]
for _ in range(steps):
    grad = gradient(pos, pos)
    sq_grad = beta * sq_grad + (1 - beta) * grad**2
    pos -= (lr / (np.sqrt(sq_grad) + 1e-8)) * grad
    path.append(pos.copy())
path = np.array(path)

x_grid = np.linspace(-9, 9, 100)
y_grid = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = loss_function(X, Y)

plt.figure(figsize=(8, 5))
plt.contour(X, Y, Z, levels=25, cmap='gray', alpha=0.4)
plt.plot(path[:, 0], path[:, 1], marker='o', markersize=4, color='purple', linewidth=2, label='Path: RMSprop')
plt.scatter(start_pos, start_pos, color='black', s=80, zorder=5, label='Start')
plt.scatter(0, 0, color='gold', edgecolor='black', marker='*', s=200, zorder=5, label='Minimum')

plt.title('4. RMSprop Optimizer', fontsize=13, fontweight='bold')
plt.xlabel('W1')
plt.ylabel('W2')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
