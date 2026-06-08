import os
import numpy as np
import torch
import matplotlib.pyplot as plt


npoints = 10
sigma = 0.01

output_path = os.path.join(
    os.path.dirname(__file__),
    "output_data",
    "Ex1_n%i_sigma%.2f_GT.npz"%(npoints, sigma)
)

data_ex1 = np.load(output_path)
for key in data_ex1:
    print(key)
    if key != "parameter_order":
        globals()[key] = torch.tensor(data_ex1[key])
used_parameters = data_ex1["parameter_order"]
print(test_x.shape)
u = mean[test_x[:,-1] == 0].reshape(51, 51)
v = mean[test_x[:,-1] == 1].reshape(51, 51)
GT = mean[test_x[:,-1] == 2].reshape(51, 51)
error = []
outputs = [u, v, GT]
error = [u-test_y[test_x[:,-1] == 0].reshape(51, 51), v-test_y[test_x[:,-1] == 1].reshape(51, 51), GT]
x = test_x[:,0][test_x[:,-1] == 2].reshape(51, 51)
y = test_x[:,1][test_x[:,-1] == 2].reshape(51, 51)

for i in range(3):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    mask = test_x[:,-1] == i
    ax.plot_surface(test_x[mask,0].reshape(51, 51), test_x[mask,1].reshape(51, 51), mean[mask].reshape(51, 51),  cmap = "jet", label='Train GT')
    ax.plot_wireframe(test_x[mask,0].reshape(51, 51), test_x[mask,1].reshape(51, 51), test_y[mask].reshape(51, 51),  cmap = "jet", label='Train GT')
    mask = train_x[:,-1] == i
    ax.scatter(train_x[mask,0], train_x[mask,1], train_y[mask],  cmap = "jet", label='Train GT',)



plt.show()