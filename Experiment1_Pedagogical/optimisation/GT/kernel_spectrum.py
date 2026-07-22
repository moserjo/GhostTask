import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch
import gpytorch
import Experiment1_Pedagogical_GT as ex
from training import train
import numpy as np
torch.set_default_dtype(torch.float64)
import time





npoints = 10
sigma = 0.2
run = 0
import matplotlib.pyplot as plt
plt.figure()



for mode in ["extra_npoints_extra_sigma", "extra_npoints"]:

    if mode == "extra_npoints_extra_sigma":
        data_path = os.path.join(
            os.path.dirname(__file__),
            "../../input_data", "inverse",
            "Ex1_n%i_sigma%.2f_GT_extra_npoints.npz"%(npoints, sigma))
    else:
        data_path = os.path.join(
            os.path.dirname(__file__),
            "../../input_data", "inverse",
            "Ex1_n%i_sigma%.2f_GT_%s.npz"%(npoints, sigma, mode))

    data = np.load(data_path)
    train_x = torch.tensor(data["train_x"], dtype=torch.float64)
    test_x = torch.tensor(data["test_x"], dtype=torch.float64)
    train_y = torch.tensor(data["train_y"], dtype=torch.float64)
    test_y = torch.tensor(data["test_y"], dtype=torch.float64)
    num_tasks = 3

    noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))
    if mode == "extra_npoints_extra_sigma":
        noise_tensor[abs(train_x[...,-1] -2)<1e-10] = 1e-6 #exactly zero

    start_time = time.time()


    modified_parameters = { "a": [2.+torch.randn(1), True, False], 
                            "lengthscale":[2.+torch.randn(1), True, gpytorch.constraints.Positive()], 
                            "amplitude":[2.+torch.randn(1), True, gpytorch.constraints.Positive()]}
                
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise_tensor)              #to check format of noise                                                                                                                 
    model = ex.PCGP_Model(train_x, train_y, likelihood, modified_parameters)


    #test how quickly the eigenvalues decay
    with torch.no_grad():
        K = model.covar_module(train_x).evaluate()
        eigvals = torch.linalg.eigvalsh(K+noise_tensor)
        print(eigvals.min())
        plt.plot(range(len(eigvals)), eigvals, "o", label = mode)
plt.yscale("log")
plt.legend()
plt.show()
