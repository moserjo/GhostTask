import sys
import os
import torch
import gpytorch
import Tripendulum_GT_forward as ex
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from training import train
torch.set_default_dtype(torch.float64)
import time


sigma= 0.001

torch.manual_seed(5)


data_path = os.path.join(
        os.path.dirname(__file__),
        "../../input_data", "forward",
        "Ex2_GT.npz")

data = np.load(data_path)
train_x = torch.tensor(data["train_x"], dtype=torch.float64)
test_x = torch.tensor(data["test_x"], dtype=torch.float64)
train_y = torch.tensor(data["train_y"], dtype=torch.float64)
test_y = torch.tensor(data["test_y"], dtype=torch.float64)
num_tasks = max(test_x[:,-1])+1
length_true = 2.5
g_true = 9.81


noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))
noise_tensor[abs(train_x[:,-1] - 4)<1e-10] = 0.001**2


modified_parameters = {  "length": [2.5, False, gpytorch.constraints.Interval(0.5, 4.)],
                "g": [9.81, False, gpytorch.constraints.Positive()],
                "amplitude": [10., True, gpytorch.constraints.Positive()], 
                "lengthscale": [1.9, True, gpytorch.constraints.GreaterThan(train_x[5,0]-train_x[1,0])],
                }


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
with torch.no_grad():#, gpytorch.settings.fast_pred_var():
        predictions = model(test_x)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

output_path = os.path.join(
    os.path.dirname(__file__),
    "../../output_data", "forward",
    "Ex2_GT.npz")

with torch.no_grad():   
    np.savez_compressed(
            output_path,
            mean=mean.numpy(),
            lower = lower.numpy(),
            upper = upper.numpy(),
            test_x = test_x.numpy(),
            train_x = train_x.numpy(),
            train_y = train_y.numpy(),
            test_y = test_y.numpy(), 
            learned_parameters = None,
            true_parameters = np.array([length_true, g_true]),
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