import radial_basis_function_extractor as rbfe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools

n_pos = 100
n_speed = 100
min_pos = -1.2
max_pos = 0.6
max_speed = 0.07
position = np.linspace(min_pos, max_pos, n_pos)
speed = np.linspace(-max_speed, max_speed, n_speed)

number_of_kernels = 10
number_of_kernels_per_dim = [10, 10]
extractor = rbfe.RadialBasisFunctionExtractor(number_of_kernels_per_dim)

states = np.asarray([x for x in itertools.product(position, speed)])

z = extractor.encode_states_with_radial_basis_functions(states)

xs, ys = np.meshgrid(position, speed)

print(np.shape(z))

feature_1 = np.reshape(z[:, -1], (n_pos, n_speed))
feature_2 = np.reshape(z[:, -2], (n_pos, n_speed))

fig_1 = plt.figure(1)
ax = fig_1.gca(projection='3d')

surf = ax.plot_surface(xs.T, ys.T, feature_1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

fig_1.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_xticks([4, 8, 12, 16, 20])
# ax.set_zticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel('', fontsize=15, rotation=150)
ax.set_ylabel('', fontsize=15)
ax.set_zlabel('', fontsize=15, rotation=60)
plt.show()


fig_2 = plt.figure(2)
ax = fig_2.gca(projection='3d')

surf = ax.plot_surface(xs.T, ys.T, feature_2, cmap=cm.coolwarm, linewidth=0, antialiased=False)

fig_2.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_xticks([4, 8, 12, 16, 20])
# ax.set_zticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel('', fontsize=15, rotation=150)
ax.set_ylabel('', fontsize=15)
ax.set_zlabel('', fontsize=15, rotation=60)
plt.show()
