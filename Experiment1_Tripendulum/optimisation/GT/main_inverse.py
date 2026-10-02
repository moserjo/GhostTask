import sys
import os
import torch
import gpytorch
import Tripendulum_GT as ex
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from training import train, peak_rss
torch.set_default_dtype(torch.float64)
import time

if len(sys.argv) != 5:
    print("Usage: python script.py <npoints> <sigma> <run> <mode>")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])
if sigma == 0:
    sigma = 0.001
sigmaGT = 0.001
run = float(sys.argv[3])
mode = sys.argv[4]
torch.manual_seed(42)

if mode == "extra_npoints_extra_sigma":
    data_path = os.path.join(
        os.path.dirname(__file__),
        "../../input_data", "extra_npoints",
        "Ex1_GT_n%i_sigma%.3f_%i.npz"%(npoints, sigma, run))
elif mode == "known_u_known_sigma":
    data_path = os.path.join(
        os.path.dirname(__file__),
        "../../input_data", "known_u",
        "Ex1_GT_n%i_sigma%.3f_%i.npz"%(npoints, sigma, run))
else:
    data_path = os.path.join(
        os.path.dirname(__file__),
        "../../input_data", mode,
        "Ex1_GT_n%i_sigma%.3f_%i.npz"%(npoints, sigma, run))

data = np.load(data_path)
train_x = torch.tensor(data["train_x"], dtype=torch.float64)
test_x = torch.tensor(data["test_x"], dtype=torch.float64)
train_y = torch.tensor(data["train_y"], dtype=torch.float64)
test_y = torch.tensor(data["test_y"], dtype=torch.float64)
num_tasks = max(test_x[:,-1])+1
length_true = 2.5
g_true = 9.81


noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))
if mode == "known_u_known_sigma":
    noise_tensor[abs(train_x[:,-1] - 4)<1e-10] = sigmaGT**2
    noise_tensor[abs(train_x[:,-1] - 3)<1e-10] = sigmaGT**2
if mode == "extra_npoints_extra_sigma":
    noise_tensor[abs(train_x[:,-1] - 4)<1e-10] = sigmaGT**2

modified_parameters = {  "length": [2.5+ torch.randn(1), True, gpytorch.constraints.Interval(0.5, 4.)],
                "g": [9.81+torch.randn(1), True, gpytorch.constraints.Positive()],
                "amplitude": [5., True, gpytorch.constraints.Positive()], 
                "lengthscale": [1.9, True, gpytorch.constraints.GreaterThan(train_x[1,0]-train_x[0,0])]}


likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise_tensor)                                  
model = ex.PCGP_Model(train_x, train_y, likelihood, modified_parameters)


mem_baseline = peak_rss()
start_time = time.time()
model.train()
likelihood.train()
training_iter = 50   #minimum number of iterations
max_iter = 7500   #cap, to prevent excessively long runs
with gpytorch.settings.max_cholesky_size(float('inf')):
    train_output = train(model,
                         likelihood,
                         modified_parameters,
                         train_x,
                         train_y,
                         training_iter=training_iter,
                         learning_rate = 0.1,
                         laplace = False,
                         convergence_parameters = ["length", "g"],
                         max_iter = max_iter)
end_time = time.time()
convergencetime = end_time - start_time
peakmem_train = peak_rss()

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model(test_x)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

peakmem_eval = peak_rss()
output_path = os.path.join(
    os.path.dirname(__file__),
    "../output_data", mode,
    "Ex1_GT_n%i_sigma%.3f_%i.npz"%(npoints, sigma, run)
)
learned_parameters = np.array([train_output.MAP_estimates[train_output.parameter_index["length"]], train_output.MAP_estimates[train_output.parameter_index["g"]]])
with torch.no_grad():   
    np.savez_compressed(
            output_path,
            mean=mean.numpy(),
            test_x = test_x.numpy(),
            train_x = train_x.numpy(),
            train_y = train_y.numpy(),
            test_y = test_y.numpy(), 
            learned_parameters = learned_parameters,
            true_parameters = np.array([length_true, g_true]),
            MAP_estimates = train_output.MAP_estimates,
            parameter_index = train_output.parameter_index,
            loss = train_output.loss,
            **train_output.parameters_during_training,
            convergencetime = convergencetime,
            n_iter = train_output.n_iter,
            converged = train_output.converged,
            max_iter_reached = train_output.max_iter_reached,
            mem_baseline = mem_baseline,
            peakmem_train = peakmem_train,
            peakmem_eval = peakmem_eval
        )   
print("saved")
print("--- %s seconds ---" % (time.time() - start_time))