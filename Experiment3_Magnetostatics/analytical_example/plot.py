
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from matplotlib import cm  # Colormaps
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_quiver_3d(train_x, train_y, title="Vector field"):
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mask = ~torch.isnan(train_y).any(dim=1)

    X = train_x[mask, 0].numpy()
    Y = train_x[mask, 1].numpy()
    Z = train_x[mask, 2].numpy()

    U = train_y[mask, 0].numpy()
    V = train_y[mask, 1].numpy()
    W = train_y[mask, 2].numpy()

    
    ax.quiver(X, Y, Z, U, V, W)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Vector field (NaNs removed)")

    


data_path = os.path.join(
    os.path.dirname(__file__),
    "output_data","Ex3_test.npz"
)
data = np.load(data_path)
for key in data:
    globals()[key] = torch.tensor(data[key])
n = 12
fig, ax = plt.subplots(1, 8, figsize = (16, 2))

test_n = n
xxx = torch.linspace(0, 1, test_n)
yyy = torch.linspace(0, 1, test_n)
zzz = torch.linspace(1, 2, test_n)
tx, ty, tz= torch.meshgrid(xxx, yyy, zzz, indexing = "ij")
z_idx = 0   # first z plane
X = tx[:, :, z_idx]
Y = ty[:, :, z_idx]


#for visualizations: 
title = ["Ax", "Ay", "Az", "J", "GT", "Bx", "By", "Bz"]
for i in range(8):
    mask = test_x[:, -1] == i

    task_max_err = max(abs(mean[mask] - test_y[mask]))
    task_max_mean = max(abs(mean[mask]))
    task_max_test = max(abs(test_y[mask]))
    print(i, task_max_err, task_max_mean, task_max_test)

    Z = mean[mask].reshape(n, n, n)[:, :, z_idx]

    cf = ax[i].contourf(X, Y, Z)
    fig.colorbar(cf, ax=ax[i])
    ax[i].set_title(title[i])
plt.tight_layout()

data_path = os.path.join(
    os.path.dirname(__file__),
    "output_data","Ex3_test_PIGP.npz"
)
data = np.load(data_path)
for key in data:
    globals()[key] = torch.tensor(data[key])
n = 12
fig, ax = plt.subplots(1, 8, figsize = (16, 2))

test_n = n
xxx = torch.linspace(0, 1, test_n)
yyy = torch.linspace(0, 1, test_n)
zzz = torch.linspace(1, 2, test_n)
tx, ty, tz= torch.meshgrid(xxx, yyy, zzz, indexing = "ij")
z_idx = 0   # first z plane
X = tx[:, :, z_idx]
Y = ty[:, :, z_idx]


#for visualizations: 
title = ["Ax", "Ay", "Az", "J", "GT", "Bx", "By", "Bz"]
for i in range(8):
    mask = test_x[:, -1] == i

    task_max_err = max(abs(mean[mask] - test_y[mask]))
    task_max_mean = max(abs(mean[mask]))
    task_max_test = max(abs(test_y[mask]))
    print(i, task_max_err, task_max_mean, task_max_test)

    Z = mean[mask].reshape(n, n, n)[:, :, z_idx]

    cf = ax[i].contourf(X, Y, Z)
    fig.colorbar(cf, ax=ax[i])
    ax[i].set_title(title[i])
plt.tight_layout()
plt.show()
"""
#train_x = torch.randn((3,3))
#train_y = torch.randn((3,3))
print(mean[:10,:])
plot_quiver_3d(test_x[:,:], mean[:,:], "title")

plt.figure()
plt.contourf(test_x[:,0].reshape(10,10,10)[:,:,5], test_x[:,1].reshape(10,10,10)[:,:,5], mean[:,2].reshape(10,10,10)[:,:,5], levels=50, cmap='viridis')
plt.colorbar(label='A3')
plt.show()"""
