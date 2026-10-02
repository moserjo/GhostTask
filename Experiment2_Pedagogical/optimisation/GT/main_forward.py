import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch
import gpytorch
import Experiment2_Pedagogical_GT as ex
from training import train, peak_rss
import numpy as np
torch.set_default_dtype(torch.float64)
import time



torch.manual_seed(1)
sigma = 0.001

data_path = os.path.join(
    os.path.dirname(__file__),
    "../../input_data", "forward",
    "Ex2_GT.npz")


data = np.load(data_path)
train_x = torch.tensor(data["train_x"], dtype=torch.float64)
test_x = torch.tensor(data["test_x"], dtype=torch.float64)
train_y = torch.tensor(data["train_y"], dtype=torch.float64)
test_y = torch.tensor(data["test_y"], dtype=torch.float64)
num_tasks = 3

noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))

mem_baseline = peak_rss()
start_time = time.time()


modified_parameters = { "a": [2., False, False], 
                        "lengthscale":[2.+torch.randn(1), True, gpytorch.constraints.Positive()], 
                        "amplitude":[2.+torch.randn(1), True, gpytorch.constraints.Positive()]}
            
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise_tensor)              #to check format of noise                                                                                                                 
model = ex.PCGP_Model(train_x, train_y, likelihood, modified_parameters)

model.train()
likelihood.train()
training_iter = 100
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
peakmem_train = peak_rss()

eval_start = time.time()
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model(test_x)
        mean = predictions.mean
evaluationtime = time.time() - eval_start
peakmem_eval = peak_rss()

output_path = os.path.join(
    os.path.dirname(__file__),
    "../output_data", "forward",
    "Ex2_GT.npz"
)

with torch.no_grad():   
    np.savez_compressed(
            output_path,
            mean=mean.numpy(),
            test_x = test_x.numpy(),
            train_x = train_x.numpy(),
            train_y = train_y.numpy(),
            test_y = test_y.numpy(), 
            learned_parameters = None,
            true_parameters = np.array([2.]),
            MAP_estimates = train_output.laplace.MAP_estimates,
            covariance = train_output.laplace.covariance.numpy(),
            hessian = train_output.laplace.hessian.numpy(),
            parameter_index = train_output.laplace.parameter_index,
            loss = train_output.loss,
            **train_output.parameters_during_training,
            trainingtime = trainingtime,
            evaluationtime = evaluationtime,
            mem_baseline = mem_baseline,
            peakmem_train = peakmem_train,
            peakmem_eval = peakmem_eval
        )   
print("saved")
print("--- %s seconds ---" % (time.time() - start_time))



