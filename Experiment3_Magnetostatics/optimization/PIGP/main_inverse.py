import sys
import os
import torch
import gpytorch
import Experiment3_PIGP as ex

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
from training import train
torch.set_default_dtype(torch.float64)
import time
import copy

if len(sys.argv) != 5:
    print("Usage: python script.py <npoints> <sigma> <run> <mode>")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])
if sigma == 0:
    sigma = 0.001
sigmaPIGP = 0.001
run = int(sys.argv[3])
mode = sys.argv[4]
torch.manual_seed(run+10)

print(f"Threads used: {torch.get_num_threads()}")
if mode == "extra_npoints_extra_sigma":
    data_path = os.path.join(os.path.dirname(__file__), "../../input_data", "extra_npoints", f"Ex3_PIGP_n{npoints}_sigma{sigma:.3f}.npz")
else:
    data_path = os.path.join(os.path.dirname(__file__), "../../input_data", mode, f"Ex3_PIGP_n{npoints}_sigma{sigma:.3f}.npz")
data = np.load(data_path)
train_x = torch.tensor(data["train_x"], dtype=torch.float64)
train_y      = torch.tensor(data["train_y"], dtype=torch.float64)
test_x       = torch.tensor(data["test_x"],  dtype=torch.float64)
test_y       = torch.tensor(data["test_y"],  dtype=torch.float64)



modified_parameters = {  "nu0": [1.+torch.randn(1), True, gpytorch.constraints.Positive()], 
                "amplitude": [5., True, gpytorch.constraints.Positive()],
                "lengthscale": [2., True, gpytorch.constraints.Positive()],}

noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))
if mode == "extra_npoints_extra_sigma":
    noise_tensor[(train_x[...,-1] == 4)] = 1e-6 #zero tasks are exactly known
    noise_tensor[(train_x[...,-1] == 5)] = 1e-6 #zero tasks are exactly known


likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise_tensor)
model = ex.PCGP_Model(train_x, train_y, likelihood, modified_parameters)

start_time = time.time()
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
with torch.no_grad(), gpytorch.settings.max_cholesky_size(float('inf')):#, gpytorch.settings.fast_pred_var():
        predictions = model(test_x)
        mean = predictions.mean


print(train_output.parameters_during_training["nu0"][-1])
output_path = os.path.join(
    os.path.dirname(__file__),
    "../../output_data", mode, 
    f"Ex3_PIGP_n{npoints}_sigma{sigma:.3f}_{run}.npz"
)


with torch.no_grad():
    np.savez_compressed(
            output_path,
            mean=mean.numpy(),
            test_x = test_x.numpy(),
            train_x = train_x.numpy(),
            train_y = train_y.numpy(),
            test_y = test_y.numpy(),

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
