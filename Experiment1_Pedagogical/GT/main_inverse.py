import sys
import os
import torch
import gpytorch
import Experiment1_Pedagogical_GT as ex
import PCGP.gpytorch_tools as gt
import numpy as np
torch.set_default_dtype(torch.float64)
import time


mode = "inverse_extra" 

if len(sys.argv) != 4:
    print("Usage: python script.py <npoints> <sigma> <run>")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])
run = float(sys.argv[3])
torch.manual_seed(run)

data_path = os.path.join(
    os.path.dirname(__file__),
    "../input_data",
    "Ex1_n%i_sigma%.2f_GT_%s.npz"%(npoints, sigma, mode))

data = np.load(data_path)
train_x = torch.tensor(data["train_x"], dtype=torch.float64)
print(train_x.shape)
test_x = torch.tensor(data["test_x"], dtype=torch.float64)
train_y = torch.tensor(data["train_y"], dtype=torch.float64)
test_y = torch.tensor(data["test_y"], dtype=torch.float64)
num_tasks = 3

start_time = time.time()
modified_parameters = { "a": [2.+torch.randn(1), True, False], 
                        "lengthscale":[.7, True, gpytorch.constraints.Positive()], 
                        "amplitude":[1., True, gpytorch.constraints.Positive()]}
            

noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))
noise_tensor[abs(train_x[:,1] -2)<1e-10] = 1e-6 #GT
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise_tensor)              #to check format of noise                                                                                                                 
model = ex.PCGP_Model(train_x, train_y, likelihood, modified_parameters)
#gt.fix_task_noises(torch.tensor([sigma, sigma, sigma])**2, model)

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
N_training = 500
train_output = gt.train(model, 
                        likelihood, 
                        modified_parameters, 
                        train_x, 
                        train_y, 
                        num_tasks,
                        noise_constraints=None, 
                        training_iter= N_training, laplace=True)

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model(test_x)
        mean = predictions.mean
      
end_time = time.time()
runtime = end_time - start_time
output_path = os.path.join(
    os.path.dirname(__file__),
    "../output_data",
    "Ex1_n%i_sigma%.2f_GT_%s_%i.npz"%(npoints, sigma, mode, run)
)



with torch.no_grad():   
    np.savez_compressed(
            output_path,
            mean=mean.numpy(),
            test_x = test_x.numpy(),
            train_x = train_x.numpy(),
            train_y = train_y.numpy(),
            test_y = test_y.numpy(), 
           
            learned_parameters = np.squeeze(np.array([train_output.parameters_during_training[key][-1] for key in ["a"]])),
            true_parameters = np.array([2.]),
            laplace_covariance = train_output.covariance_matrix.numpy(),
            hessian = train_output.hessian.numpy(),
            loss = train_output.loss_landscape,
            #test_loss = train_output.test_loss.numpy(),
            **train_output.parameters_during_training,
            runtime = runtime
        )   
print("saved")
print("--- %s seconds ---" % (time.time() - start_time))



