import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch
import gpytorch
import PIGP.Experiment1_Pedagogical_PIGP as ex
import GT.Experiment1_Pedagogical_GT as ex2
from training import train
import numpy as np
torch.set_default_dtype(torch.float64)
import time





npoints = 5
sigma = 0.2
run = 0
import matplotlib.pyplot as plt
plt.figure()



for data_mode in ["extra_npoints_extra_sigma", "extra_npoints"]:

    if data_mode == "extra_npoints_extra_sigma":
        data_path = os.path.join(
            os.path.dirname(__file__),
            "../input_data", "inverse",
            "Ex1_n%i_sigma%.2f_PIGP_extra_npoints.npz"%(npoints, sigma))
    else:
        data_path = os.path.join(
            os.path.dirname(__file__),
            "../input_data", "inverse",
            "Ex1_n%i_sigma%.2f_PIGP_%s.npz"%(npoints, sigma, data_mode))

    data = np.load(data_path)
    train_x = torch.tensor(data["train_x"], dtype=torch.float64)
    test_x = torch.tensor(data["test_x"], dtype=torch.float64)
    train_y = torch.tensor(data["train_y"], dtype=torch.float64)
    test_y = torch.tensor(data["test_y"], dtype=torch.float64)
    num_tasks = 4

    noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))
    if data_mode == "extra_npoints_extra_sigma":
        noise_tensor[abs(train_x[...,-1] -2)<1e-10] = 1e-6 #exactly zero
        noise_tensor[abs(train_x[...,-1] -3)<1e-10] = 1e-6 #exactly zero

    start_time = time.time()


    modified_parameters = { "a": [2., False, False], 
                                "lengthscale":[2., True, gpytorch.constraints.Positive()], 
                                "amplitude":[2., True, gpytorch.constraints.Positive()]}
                
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise_tensor)              #to check format of noise                                                                                                                 
    model = ex.PCGP_Model(train_x, train_y, likelihood, modified_parameters)


    #test how quickly the eigenvalues decay
    with torch.no_grad():
        K = model.covar_module(train_x).evaluate()
        eigvals = torch.linalg.eigvalsh(K+torch.diag(noise_tensor))
        eigvals = torch.flip(eigvals, dims=[0])  # descending: largest at x=0
        print(data_mode, "PIGP")
        print(eigvals.max()/eigvals.min())
        plt.plot(range(len(eigvals)), abs(eigvals), "o", label = data_mode)



for data_mode in ["extra_npoints_extra_sigma", "extra_npoints"]:

    if data_mode == "extra_npoints_extra_sigma":
        data_path = os.path.join(
            os.path.dirname(__file__),
            "../input_data", "inverse",
            "Ex1_n%i_sigma%.2f_GT_extra_npoints.npz"%(npoints, sigma))
    else:
        data_path = os.path.join(
            os.path.dirname(__file__),
            "../input_data", "inverse",
            "Ex1_n%i_sigma%.2f_GT_%s.npz"%(npoints, sigma, data_mode))

    data = np.load(data_path)
    train_x = torch.tensor(data["train_x"], dtype=torch.float64)
    test_x = torch.tensor(data["test_x"], dtype=torch.float64)
    train_y = torch.tensor(data["train_y"], dtype=torch.float64)
    test_y = torch.tensor(data["test_y"], dtype=torch.float64)
    num_tasks = 3

    noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))
    if data_mode == "extra_npoints_extra_sigma":
        print("here")
        noise_tensor[abs(train_x[...,-1] -2)<1e-10] = 1e-6 #exactly zero

    start_time = time.time()


    modified_parameters = { "a": [2., False, False], 
                            "lengthscale":[2., True, gpytorch.constraints.Positive()], 
                            "amplitude":[2., True, gpytorch.constraints.Positive()]}
                
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise_tensor)              #to check format of noise                                                                                                                 
    model = ex2.PCGP_Model(train_x, train_y, likelihood, modified_parameters)


    #test how quickly the eigenvalues decay
    with torch.no_grad():
        K = model.covar_module(train_x).evaluate()
        eigvals = torch.linalg.eigvalsh(K+torch.diag(noise_tensor))
        eigvals = torch.flip(eigvals, dims=[0])  # descending: largest at x=0
        print(data_mode, "GT")
        print(eigvals.max()/eigvals.min())
        plt.plot(range(len(eigvals)), abs(eigvals), "x", label = data_mode)
plt.yscale("log")
plt.legend()
plt.show()
