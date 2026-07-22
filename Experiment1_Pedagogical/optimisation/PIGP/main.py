import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch
import gpytorch
import Experiment1_Pedagogical_PIGP as ex
from training import train
import numpy as np
torch.set_default_dtype(torch.float64)
import time



if len(sys.argv) != 5:
    print("Usage: python script.py <npoints> <sigma> <run> <mode>")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])
run = float(sys.argv[3])
mode = sys.argv[4]
torch.manual_seed(run)


if mode == "extra_npoints_extra_sigma":
    data_path = os.path.join(
        os.path.dirname(__file__),
        "../../input_data", "inverse",
        "Ex1_n%i_sigma%.2f_PIGP_extra_npoints.npz"%(npoints, sigma))
else:
    data_path = os.path.join(
        os.path.dirname(__file__),
        "../input_data", "inverse",
        "Ex1_n%i_sigma%.2f_PIGP_%s.npz"%(npoints, sigma, mode))

data = np.load(data_path)
train_x = torch.tensor(data["train_x"], dtype=torch.float64)
test_x = torch.tensor(data["test_x"], dtype=torch.float64)
train_y = torch.tensor(data["train_y"], dtype=torch.float64)
test_y = torch.tensor(data["test_y"], dtype=torch.float64)
num_tasks = 4

noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))
if mode == "extra_npoints_extra_sigma":
    noise_tensor[abs(train_x[...,-1] -2)<1e-10] = 1e-6 #exactly zero
    noise_tensor[abs(train_x[...,-1] -3)<1e-10] = 1e-6 #exactly zero

start_time = time.time()


modified_parameters = { "a": [2.+torch.randn(1), True, False], 
                        "lengthscale":[2.+torch.randn(1), True, gpytorch.constraints.Positive()], 
                        "amplitude":[2.+torch.randn(1), True, gpytorch.constraints.Positive()]}
            
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise_tensor)              #to check format of noise                                                                                                                 
model = ex.PCGP_Model(train_x, train_y, likelihood, modified_parameters)

"""
#test how quickly the eigenvalues decay
with torch.no_grad():
    K = model.covar_module(train_x).evaluate()
    eigvals = torch.linalg.eigvalsh(K+sigma**2*torch.eye(K.shape[0]))
    print(eigvals.min())
    import matplotlib.pyplot as plt
    plt.plot(range(len(eigvals)), eigvals, "o")
    plt.yscale("log")
    plt.show()
"""

model.train()
likelihood.train()
training_iter = 500
with gpytorch.settings.max_cholesky_size(float('inf')):
    train_output = train(model,
                         likelihood,
                         modified_parameters,
                         train_x,
                         train_y,
                         training_iter=training_iter,
                         learning_rate = 0.1,
                         laplace = True)
end_time = time.time()
trainingtime = end_time - start_time

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model(test_x)
        mean = predictions.mean
output_path = os.path.join(
    os.path.dirname(__file__),
    "../output_data", mode, 
    "Ex1_n%i_sigma%.2f_PIGP_%i.npz"%(npoints, sigma, run)
)


learned_parameters = train_output.laplace.MAP_estimates[train_output.laplace.parameter_index["a"]]
with torch.no_grad():   
    np.savez_compressed(
            output_path,
            mean=mean.numpy(),
            test_x = test_x.numpy(),
            train_x = train_x.numpy(),
            train_y = train_y.numpy(),
            test_y = test_y.numpy(), 
            learned_parameters = learned_parameters,
            true_parameters = np.array([2.]),
            MAP_estimates = train_output.laplace.MAP_estimates,
            covariance = train_output.laplace.covariance.numpy(),
            hessian = train_output.laplace.hessian.numpy(),
            parameter_index = train_output.laplace.parameter_index,
            loss = train_output.loss,
            **train_output.parameters_during_training,
            trainingtime = trainingtime
        )   
print("saved")
print("--- %s seconds ---" % (time.time() - start_time))



