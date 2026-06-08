import sys
import os
import torch
import gpytorch
import Experiment3_analytic_PIGP as ex
import PCGP.gpytorch_tools as gt
import numpy as np
torch.set_default_dtype(torch.float64)
import time
import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401



n = 6
x = torch.linspace(0, 1, n)
y = torch.linspace(0, 1, n)
z = torch.linspace(1, 2, n)
full_grid = torch.cartesian_prod(x, y, z)
xx = full_grid[:,0]
yy = full_grid[:,1]
zz = full_grid[:,2]

train_x_0 = full_grid[((full_grid[:, 1] == 0) |(full_grid[:, 1] == 1) |(full_grid[:, 2] == 1) |(full_grid[:, 2] == 2))]
train_x_1 = full_grid[((full_grid[:, 0] == 0) |(full_grid[:, 0] == 1) |(full_grid[:, 2] == 1) |(full_grid[:, 2] == 2))]
train_x_2 = full_grid[((full_grid[:, 0] == 0) |(full_grid[:, 0] == 1) |(full_grid[:, 1] == 0) |(full_grid[:, 1] == 1))]
Jz = 2*torch.pi**2*torch.sin(torch.pi*xx)*torch.sin(torch.pi*yy)#torch.ones_like(full_grid)
GT = torch.zeros_like(full_grid[:,0])
train_i_0 = torch.zeros_like(train_x_0[:,0])
train_i_1 = torch.ones_like(train_x_1[:,0])
train_i_2 = torch.ones_like(train_x_2[:,0])*2
train_i_Jx = torch.ones_like(full_grid[:,0])*3
train_i_Jy = torch.ones_like(full_grid[:,0])*4
train_i_Jz = torch.ones_like(full_grid[:,0])*5
train_x = torch.cat([train_x_0, train_x_1, train_x_2, full_grid, full_grid, full_grid], dim = 0)
train_i = torch.cat([train_i_0, train_i_1, train_i_2, train_i_Jx, train_i_Jy, train_i_Jz], dim = 0)
full_train_x = torch.stack([train_x[:,0], train_x[:,1], train_x[:,2], train_i], dim = -1)
train_y_0 = torch.zeros_like(train_x_0[:,0])
train_y_1 = torch.zeros_like(train_x_1[:,0])
train_y_2 = torch.zeros_like(train_x_2[:,0])
train_y_3 = GT
train_y_4 = GT
train_y_5 = Jz

train_y = torch.cat([train_y_0, train_y_1, train_y_2, train_y_3, train_y_4, train_y_5], dim = 0)
                                 

# --- helper function ---
def plot_3d_scatter(train_x, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        train_x[:, 0].numpy(),
        train_x[:, 1].numpy(),
        train_x[:, 2].numpy(),
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")



# --- plots ---
#plot_3d_scatter(train_x_0, "A1 boundary points")
#plot_3d_scatter(train_x_1, "A2 boundary points")
#plot_3d_scatter(train_x_2, "A3 boundary points")
#plot_3d_scatter(full_grid, "full grid points")
#plt.show()




test_n = 2*n
xxx = torch.linspace(0, 1, test_n)
yyy = torch.linspace(0, 1, test_n)
zzz = torch.linspace(1, 2, test_n)
test_grid = torch.cartesian_prod(xxx, yyy, zzz)
tx = test_grid[:,0]
ty = test_grid[:,1]
tz = test_grid[:,2]
test_grid_i = torch.ones_like(test_grid[:,0])
test_x = torch.cat([torch.stack([test_grid[:,0], test_grid[:,1], test_grid[:,2], i*test_grid_i], dim = -1) for i in range(0, 9)], dim = 0)
Ax = torch.zeros_like(tz)
Ay = torch.zeros_like(tz)
Az = torch.sin(torch.pi*tx)*torch.sin(torch.pi*ty)
Jz_test = 2*torch.pi**2*torch.sin(torch.pi*tx)*torch.sin(torch.pi*ty)
GT_test = torch.zeros_like(tz)
Bx = torch.pi*torch.cos(torch.pi*ty)*torch.sin(torch.pi*tx)
By = -torch.pi*torch.cos(torch.pi*tx)*torch.sin(torch.pi*ty)
Bz = torch.zeros_like(tz)

test_y = torch.cat([Ax, Ay, Az, GT_test, GT_test, Jz_test,  Bx, By, Bz], dim = 0)


sigma = 0.1


parameters = {  "nu0": [1., False, gpytorch.constraints.Positive()],
                "amplitude": [5., True, gpytorch.constraints.Positive()], 
                "lengthscale": [2., True, gpytorch.constraints.Positive()],}

noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))
#noise_tensor[(train_x[:,1] == 5)] = 1e-6 #GT#
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise_tensor)                                  
model = ex.PCGP_Model(full_train_x, train_y, likelihood, parameters)



start_time = time.time()
model.train()
likelihood.train()
N_training = 100
train_output = gt.train(model, 
                        likelihood, 
                        parameters, 
                        full_train_x, 
                        train_y, 
                        None,
                        noise_constraints=None, 
                        training_iter= N_training, laplace = True)


model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model(test_x)
        mean = predictions.mean

end_time = time.time()
runtime = end_time - start_time

output_path = os.path.join(
    os.path.dirname(__file__),
    "output_data", 
    "Ex3_test_PIGP_%.2f.npz"%sigma
)


with torch.no_grad():   
    np.savez_compressed(
            output_path,
            mean=mean.numpy(),
            test_x = test_x.numpy(),
            train_x = train_x.numpy(),
            train_y = train_y.numpy(),
            test_y = test_y.numpy(), 
           
            learned_parameters = np.squeeze(np.array([train_output.parameters_during_training[key][-1] for key in ["nu0"]])),
            true_parameters = np.array([2.]),
            laplace_covariance = train_output.covariance_matrix.numpy(),
            hessian = train_output.hessian.numpy(),
            loss = train_output.loss_landscape,
            **train_output.parameters_during_training,
            runtime = runtime
        )   
print("saved")
print("--- %s seconds ---" % (time.time() - start_time))

