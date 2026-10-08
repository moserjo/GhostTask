import sys
import os
import torch
import gpytorch
import Tripendulum_reference as ex
import numpy as np
torch.set_default_dtype(torch.float64)
import time
from training import train




npoints = 3
sigma = 0.001
run = 1
Al_mode = "withAl"  # "withAl" or "fixedAl"
torch.manual_seed(100+run**2+1)


data_path = os.path.join(
    os.path.dirname(__file__),
    "../Experiment1_Tripendulum/input_data/extra_npoints", 
    "Ex1_GT_n%i_sigma%.3f_0.npz"%(npoints, sigma)
)
data = np.load(data_path)

train_x = torch.tensor(data["train_x"])[:-25, :]
test_x = torch.tensor(data["test_x"])[:-201, :]
train_y = torch.tensor(data["train_y"])[:-25]
test_y = torch.tensor(data["test_y"])[:-201]


start_time = time.time()




num_tasks = max(test_x[:1])+1
length_true = 2.5
g_true = 9.81



if Al_mode == "withAl":
    withAl = True
else:
    withAl = False

if sigma == 0:
    sigma = 0.001
sigmaGT = 0.001

parameters = {  "length": [2.5+ torch.randn(1), True, gpytorch.constraints.Interval(0.5, 4.)],
                "g": [9.81+torch.randn(1), True, gpytorch.constraints.Positive()],
                "amplitude": [5., withAl, gpytorch.constraints.Positive()], 
                "lengthscale": [1.9, withAl, gpytorch.constraints.GreaterThan(train_x[1,0]-train_x[0,0])]}


noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))
noise_tensor[(train_x[:,1] == 5)] = 1e-6 #GT
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise_tensor)                                  
#likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
#likelihood.noise = sigma**2
#likelihood.raw_noise.requires_grad = False
model = ex.PCGP_Model(train_x, train_y, likelihood, parameters)
#gt.fix_task_noises(torch.tensor([sigma, sigma, sigma, sigma, sigmaGT])**2, model)

model.train()
likelihood.train()
N_training = 1500
training_iter = 1500
with gpytorch.settings.max_cholesky_size(float('inf')):
    train_output = train(model,
                         likelihood,
                         parameters,
                         train_x,
                         train_y,
                         training_iter=training_iter,
                         learning_rate = 0.1,
                         laplace = True)


model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model(test_x)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

end_time = time.time()
runtime = end_time - start_time

output_path = os.path.join(
    os.path.dirname(__file__),
    "output",
    "reference_n%i_sigma%.2f.npz"%(npoints, sigma)
)
for key in train_output.parameters_during_training:
    print(train_output.parameters_during_training[key].shape, key)
with torch.no_grad():   
    np.savez_compressed(
            output_path,
            mean=mean.numpy(),
            test_x = test_x.numpy(),
            train_x = train_x.numpy(),
            train_y = train_y.numpy(),
            test_y = test_y.numpy(), 
            lower=lower.numpy(),
            upper = upper.numpy(),
            learned_parameters = np.squeeze(np.array([train_output.parameters_during_training[key][-1] for key in ["length", "g"]])),
            true_parameters = np.array([length_true, g_true]),
            #laplace_covariance = train_output.covariance_matrix.numpy(),
            #hessian = train_output.hessian.numpy(),
            #loss = train_output.loss_landscape,
            #test_loss = train_output.test_loss.numpy(),
            **train_output.parameters_during_training,
            runtime = runtime
        )   
print("saved")
print("--- %s seconds ---" % (time.time() - start_time))
